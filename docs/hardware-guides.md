---
title: Hardware Setup Guides
sidebar_position: 100
---

# Hardware Setup Guides

## Overview

This section contains hardware setup guides for all modules in the Physical AI & Humanoid Robotics course. Each module has specific hardware requirements with detailed setup instructions.

## Module-Specific Hardware Requirements

### ROS 2 Module Hardware
- **Computing Platform**: Computer with 8GB RAM minimum (16GB+ recommended)
- **Operating System**: Ubuntu 22.04 LTS or Windows 10/11 with WSL2
- **Development Environment**: ROS 2 Humble Hawksbill
- **Additional Requirements**: USB ports for connecting robot hardware

### Gazebo/Unity Module Hardware
- **Graphics Card**: NVIDIA GPU (RTX series recommended) for advanced simulation
- **RAM**: 16GB+ for complex simulations
- **Processor**: Multi-core processor (Quad-core minimum)

### NVIDIA Isaac Module Hardware
- **NVIDIA Jetson Development Kit** (Xavier NX, Orin, or equivalent)
- **Robot Platform**: ROS 2 compatible robot (e.g., TurtleBot3, Clearpath platforms)
- **Sensors**: RGB-D camera, LiDAR, IMU as specified in lab exercises

### VLA Module Hardware
- **High-performance computing**: RTX workstation for vision-language processing
- **Specialized sensors**: Intel RealSense, or equivalent depth camera
- **Robotic platforms**: NVIDIA Jetson-based robots with AI capabilities

## General Setup Instructions

Each module contains detailed hardware setup instructions in its respective lab exercises. Follow the specific guide for the module you're working on:

- [ROS 2 Hardware Setup](./ros2/week1-2.md)
- [Gazebo/Unity Hardware Setup](./gazebo-unity/week4-5.md)
- [NVIDIA Isaac Hardware Setup](./nvidia-isaac/week7-8.md)
- [VLA Hardware Setup](./vla/week10-11.md)

## Troubleshooting

For common hardware issues, see the troubleshooting section in each module's conclusion document.