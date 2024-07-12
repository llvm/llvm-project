// RUN: llvm-mc -triple i686-unknown-unknown -mattr=+avxvnni --show-encoding < %s  | FileCheck %s

// CHECK: {vex} vpdpbusd %ymm4, %ymm5, %ymm6
// CHECK: encoding: [0xc4,0xe2,0x55,0x50,0xf4]
          {vex} vpdpbusd %ymm4, %ymm5, %ymm6

// CHECK: {vex} vpdpbusd %xmm4, %xmm5, %xmm6
// CHECK: encoding: [0xc4,0xe2,0x51,0x50,0xf4]
          {vex} vpdpbusd %xmm4, %xmm5, %xmm6

// CHECK: {vex} vpdpbusd  268435456(%esp,%esi,8), %ymm5, %ymm6
// CHECK: encoding: [0xc4,0xe2,0x55,0x50,0xb4,0xf4,0x00,0x00,0x00,0x10]
          {vex} vpdpbusd  268435456(%esp,%esi,8), %ymm5, %ymm6

// CHECK: {vex} vpdpbusd  291(%edi,%eax,4), %ymm5, %ymm6
// CHECK: encoding: [0xc4,0xe2,0x55,0x50,0xb4,0x87,0x23,0x01,0x00,0x00]
          {vex} vpdpbusd  291(%edi,%eax,4), %ymm5, %ymm6

// CHECK: {vex} vpdpbusd  (%eax), %ymm5, %ymm6
// CHECK: encoding: [0xc4,0xe2,0x55,0x50,0x30]
          {vex} vpdpbusd  (%eax), %ymm5, %ymm6

// CHECK: {vex} vpdpbusd  -1024(,%ebp,2), %ymm5, %ymm6
// CHECK: encoding: [0xc4,0xe2,0x55,0x50,0x34,0x6d,0x00,0xfc,0xff,0xff]
          {vex} vpdpbusd  -1024(,%ebp,2), %ymm5, %ymm6

// CHECK: {vex} vpdpbusd  4064(%ecx), %ymm5, %ymm6
// CHECK: encoding: [0xc4,0xe2,0x55,0x50,0xb1,0xe0,0x0f,0x00,0x00]
          {vex} vpdpbusd  4064(%ecx), %ymm5, %ymm6

// CHECK: {vex} vpdpbusd  -4096(%edx), %ymm5, %ymm6
// CHECK: encoding: [0xc4,0xe2,0x55,0x50,0xb2,0x00,0xf0,0xff,0xff]
          {vex} vpdpbusd  -4096(%edx), %ymm5, %ymm6

// CHECK: {vex} vpdpbusd  268435456(%esp,%esi,8), %xmm5, %xmm6
// CHECK: encoding: [0xc4,0xe2,0x51,0x50,0xb4,0xf4,0x00,0x00,0x00,0x10]
          {vex} vpdpbusd  268435456(%esp,%esi,8), %xmm5, %xmm6

// CHECK: {vex} vpdpbusd  291(%edi,%eax,4), %xmm5, %xmm6
// CHECK: encoding: [0xc4,0xe2,0x51,0x50,0xb4,0x87,0x23,0x01,0x00,0x00]
          {vex} vpdpbusd  291(%edi,%eax,4), %xmm5, %xmm6

// CHECK: {vex} vpdpbusd  (%eax), %xmm5, %xmm6
// CHECK: encoding: [0xc4,0xe2,0x51,0x50,0x30]
          {vex} vpdpbusd  (%eax), %xmm5, %xmm6

// CHECK: {vex} vpdpbusd  -512(,%ebp,2), %xmm5, %xmm6
// CHECK: encoding: [0xc4,0xe2,0x51,0x50,0x34,0x6d,0x00,0xfe,0xff,0xff]
          {vex} vpdpbusd  -512(,%ebp,2), %xmm5, %xmm6

// CHECK: {vex} vpdpbusd  2032(%ecx), %xmm5, %xmm6
// CHECK: encoding: [0xc4,0xe2,0x51,0x50,0xb1,0xf0,0x07,0x00,0x00]
          {vex} vpdpbusd  2032(%ecx), %xmm5, %xmm6

// CHECK: {vex} vpdpbusd  -2048(%edx), %xmm5, %xmm6
// CHECK: encoding: [0xc4,0xe2,0x51,0x50,0xb2,0x00,0xf8,0xff,0xff]
          {vex} vpdpbusd  -2048(%edx), %xmm5, %xmm6

// CHECK: {vex} vpdpbusds %ymm4, %ymm5, %ymm6
// CHECK: encoding: [0xc4,0xe2,0x55,0x51,0xf4]
          {vex} vpdpbusds %ymm4, %ymm5, %ymm6

// CHECK: {vex} vpdpbusds %xmm4, %xmm5, %xmm6
// CHECK: encoding: [0xc4,0xe2,0x51,0x51,0xf4]
          {vex} vpdpbusds %xmm4, %xmm5, %xmm6

// CHECK: {vex} vpdpbusds  268435456(%esp,%esi,8), %ymm5, %ymm6
// CHECK: encoding: [0xc4,0xe2,0x55,0x51,0xb4,0xf4,0x00,0x00,0x00,0x10]
          {vex} vpdpbusds  268435456(%esp,%esi,8), %ymm5, %ymm6

// CHECK: {vex} vpdpbusds  291(%edi,%eax,4), %ymm5, %ymm6
// CHECK: encoding: [0xc4,0xe2,0x55,0x51,0xb4,0x87,0x23,0x01,0x00,0x00]
          {vex} vpdpbusds  291(%edi,%eax,4), %ymm5, %ymm6

// CHECK: {vex} vpdpbusds  (%eax), %ymm5, %ymm6
// CHECK: encoding: [0xc4,0xe2,0x55,0x51,0x30]
          {vex} vpdpbusds  (%eax), %ymm5, %ymm6

// CHECK: {vex} vpdpbusds  -1024(,%ebp,2), %ymm5, %ymm6
// CHECK: encoding: [0xc4,0xe2,0x55,0x51,0x34,0x6d,0x00,0xfc,0xff,0xff]
          {vex} vpdpbusds  -1024(,%ebp,2), %ymm5, %ymm6

// CHECK: {vex} vpdpbusds  4064(%ecx), %ymm5, %ymm6
// CHECK: encoding: [0xc4,0xe2,0x55,0x51,0xb1,0xe0,0x0f,0x00,0x00]
          {vex} vpdpbusds  4064(%ecx), %ymm5, %ymm6

// CHECK: {vex} vpdpbusds  -4096(%edx), %ymm5, %ymm6
// CHECK: encoding: [0xc4,0xe2,0x55,0x51,0xb2,0x00,0xf0,0xff,0xff]
          {vex} vpdpbusds  -4096(%edx), %ymm5, %ymm6

// CHECK: {vex} vpdpbusds  268435456(%esp,%esi,8), %xmm5, %xmm6
// CHECK: encoding: [0xc4,0xe2,0x51,0x51,0xb4,0xf4,0x00,0x00,0x00,0x10]
          {vex} vpdpbusds  268435456(%esp,%esi,8), %xmm5, %xmm6

// CHECK: {vex} vpdpbusds  291(%edi,%eax,4), %xmm5, %xmm6
// CHECK: encoding: [0xc4,0xe2,0x51,0x51,0xb4,0x87,0x23,0x01,0x00,0x00]
          {vex} vpdpbusds  291(%edi,%eax,4), %xmm5, %xmm6

// CHECK: {vex} vpdpbusds  (%eax), %xmm5, %xmm6
// CHECK: encoding: [0xc4,0xe2,0x51,0x51,0x30]
          {vex} vpdpbusds  (%eax), %xmm5, %xmm6

// CHECK: {vex} vpdpbusds  -512(,%ebp,2), %xmm5, %xmm6
// CHECK: encoding: [0xc4,0xe2,0x51,0x51,0x34,0x6d,0x00,0xfe,0xff,0xff]
          {vex} vpdpbusds  -512(,%ebp,2), %xmm5, %xmm6

// CHECK: {vex} vpdpbusds  2032(%ecx), %xmm5, %xmm6
// CHECK: encoding: [0xc4,0xe2,0x51,0x51,0xb1,0xf0,0x07,0x00,0x00]
          {vex} vpdpbusds  2032(%ecx), %xmm5, %xmm6

// CHECK: {vex} vpdpbusds  -2048(%edx), %xmm5, %xmm6
// CHECK: encoding: [0xc4,0xe2,0x51,0x51,0xb2,0x00,0xf8,0xff,0xff]
          {vex} vpdpbusds  -2048(%edx), %xmm5, %xmm6

// CHECK: vpdpwssd %ymm4, %ymm5, %ymm6
// CHECK: encoding: [0xc4,0xe2,0x55,0x52,0xf4]
          {vex} vpdpwssd %ymm4, %ymm5, %ymm6

// CHECK: vpdpwssd %xmm4, %xmm5, %xmm6
// CHECK: encoding: [0xc4,0xe2,0x51,0x52,0xf4]
          {vex} vpdpwssd %xmm4, %xmm5, %xmm6

// CHECK: vpdpwssd  268435456(%esp,%esi,8), %ymm5, %ymm6
// CHECK: encoding: [0xc4,0xe2,0x55,0x52,0xb4,0xf4,0x00,0x00,0x00,0x10]
          {vex} vpdpwssd  268435456(%esp,%esi,8), %ymm5, %ymm6

// CHECK: vpdpwssd  291(%edi,%eax,4), %ymm5, %ymm6
// CHECK: encoding: [0xc4,0xe2,0x55,0x52,0xb4,0x87,0x23,0x01,0x00,0x00]
          {vex} vpdpwssd  291(%edi,%eax,4), %ymm5, %ymm6

// CHECK: vpdpwssd  (%eax), %ymm5, %ymm6
// CHECK: encoding: [0xc4,0xe2,0x55,0x52,0x30]
          {vex} vpdpwssd  (%eax), %ymm5, %ymm6

// CHECK: vpdpwssd  -1024(,%ebp,2), %ymm5, %ymm6
// CHECK: encoding: [0xc4,0xe2,0x55,0x52,0x34,0x6d,0x00,0xfc,0xff,0xff]
          {vex} vpdpwssd  -1024(,%ebp,2), %ymm5, %ymm6

// CHECK: vpdpwssd  4064(%ecx), %ymm5, %ymm6
// CHECK: encoding: [0xc4,0xe2,0x55,0x52,0xb1,0xe0,0x0f,0x00,0x00]
          {vex} vpdpwssd  4064(%ecx), %ymm5, %ymm6

// CHECK: vpdpwssd  -4096(%edx), %ymm5, %ymm6
// CHECK: encoding: [0xc4,0xe2,0x55,0x52,0xb2,0x00,0xf0,0xff,0xff]
          {vex} vpdpwssd  -4096(%edx), %ymm5, %ymm6

// CHECK: vpdpwssd  268435456(%esp,%esi,8), %xmm5, %xmm6
// CHECK: encoding: [0xc4,0xe2,0x51,0x52,0xb4,0xf4,0x00,0x00,0x00,0x10]
          {vex} vpdpwssd  268435456(%esp,%esi,8), %xmm5, %xmm6

// CHECK: vpdpwssd  291(%edi,%eax,4), %xmm5, %xmm6
// CHECK: encoding: [0xc4,0xe2,0x51,0x52,0xb4,0x87,0x23,0x01,0x00,0x00]
          {vex} vpdpwssd  291(%edi,%eax,4), %xmm5, %xmm6

// CHECK: vpdpwssd  (%eax), %xmm5, %xmm6
// CHECK: encoding: [0xc4,0xe2,0x51,0x52,0x30]
          {vex} vpdpwssd  (%eax), %xmm5, %xmm6

// CHECK: vpdpwssd  -512(,%ebp,2), %xmm5, %xmm6
// CHECK: encoding: [0xc4,0xe2,0x51,0x52,0x34,0x6d,0x00,0xfe,0xff,0xff]
          {vex} vpdpwssd  -512(,%ebp,2), %xmm5, %xmm6

// CHECK: vpdpwssd  2032(%ecx), %xmm5, %xmm6
// CHECK: encoding: [0xc4,0xe2,0x51,0x52,0xb1,0xf0,0x07,0x00,0x00]
          {vex} vpdpwssd  2032(%ecx), %xmm5, %xmm6

// CHECK: vpdpwssd  -2048(%edx), %xmm5, %xmm6
// CHECK: encoding: [0xc4,0xe2,0x51,0x52,0xb2,0x00,0xf8,0xff,0xff]
          {vex} vpdpwssd  -2048(%edx), %xmm5, %xmm6

// CHECK: vpdpwssds %ymm4, %ymm5, %ymm6
// CHECK: encoding: [0xc4,0xe2,0x55,0x53,0xf4]
          {vex} vpdpwssds %ymm4, %ymm5, %ymm6

// CHECK: vpdpwssds %xmm4, %xmm5, %xmm6
// CHECK: encoding: [0xc4,0xe2,0x51,0x53,0xf4]
          {vex} vpdpwssds %xmm4, %xmm5, %xmm6

// CHECK: vpdpwssds  268435456(%esp,%esi,8), %ymm5, %ymm6
// CHECK: encoding: [0xc4,0xe2,0x55,0x53,0xb4,0xf4,0x00,0x00,0x00,0x10]
          {vex} vpdpwssds  268435456(%esp,%esi,8), %ymm5, %ymm6

// CHECK: vpdpwssds  291(%edi,%eax,4), %ymm5, %ymm6
// CHECK: encoding: [0xc4,0xe2,0x55,0x53,0xb4,0x87,0x23,0x01,0x00,0x00]
          {vex} vpdpwssds  291(%edi,%eax,4), %ymm5, %ymm6

// CHECK: vpdpwssds  (%eax), %ymm5, %ymm6
// CHECK: encoding: [0xc4,0xe2,0x55,0x53,0x30]
          {vex} vpdpwssds  (%eax), %ymm5, %ymm6

// CHECK: vpdpwssds  -1024(,%ebp,2), %ymm5, %ymm6
// CHECK: encoding: [0xc4,0xe2,0x55,0x53,0x34,0x6d,0x00,0xfc,0xff,0xff]
          {vex} vpdpwssds  -1024(,%ebp,2), %ymm5, %ymm6

// CHECK: vpdpwssds  4064(%ecx), %ymm5, %ymm6
// CHECK: encoding: [0xc4,0xe2,0x55,0x53,0xb1,0xe0,0x0f,0x00,0x00]
          {vex} vpdpwssds  4064(%ecx), %ymm5, %ymm6

// CHECK: vpdpwssds  -4096(%edx), %ymm5, %ymm6
// CHECK: encoding: [0xc4,0xe2,0x55,0x53,0xb2,0x00,0xf0,0xff,0xff]
          {vex} vpdpwssds  -4096(%edx), %ymm5, %ymm6

// CHECK: vpdpwssds  268435456(%esp,%esi,8), %xmm5, %xmm6
// CHECK: encoding: [0xc4,0xe2,0x51,0x53,0xb4,0xf4,0x00,0x00,0x00,0x10]
          {vex} vpdpwssds  268435456(%esp,%esi,8), %xmm5, %xmm6

// CHECK: vpdpwssds  291(%edi,%eax,4), %xmm5, %xmm6
// CHECK: encoding: [0xc4,0xe2,0x51,0x53,0xb4,0x87,0x23,0x01,0x00,0x00]
          {vex} vpdpwssds  291(%edi,%eax,4), %xmm5, %xmm6

// CHECK: vpdpwssds  (%eax), %xmm5, %xmm6
// CHECK: encoding: [0xc4,0xe2,0x51,0x53,0x30]
          {vex} vpdpwssds  (%eax), %xmm5, %xmm6

// CHECK: vpdpwssds  -512(,%ebp,2), %xmm5, %xmm6
// CHECK: encoding: [0xc4,0xe2,0x51,0x53,0x34,0x6d,0x00,0xfe,0xff,0xff]
          {vex} vpdpwssds  -512(,%ebp,2), %xmm5, %xmm6

// CHECK: vpdpwssds  2032(%ecx), %xmm5, %xmm6
// CHECK: encoding: [0xc4,0xe2,0x51,0x53,0xb1,0xf0,0x07,0x00,0x00]
          {vex} vpdpwssds  2032(%ecx), %xmm5, %xmm6

// CHECK: vpdpwssds  -2048(%edx), %xmm5, %xmm6
// CHECK: encoding: [0xc4,0xe2,0x51,0x53,0xb2,0x00,0xf8,0xff,0xff]
          {vex} vpdpwssds  -2048(%edx), %xmm5, %xmm6

