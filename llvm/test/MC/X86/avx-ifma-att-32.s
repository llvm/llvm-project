// RUN: llvm-mc -triple i686-unknown-unknown -mattr=+avxifma --show-encoding %s | FileCheck %s

// CHECK: {vex} vpmadd52huq %ymm4, %ymm3, %ymm2
// CHECK: encoding: [0xc4,0xe2,0xe5,0xb5,0xd4]
     {vex} vpmadd52huq %ymm4, %ymm3, %ymm2

// CHECK: {vex} vpmadd52huq %xmm4, %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0xe1,0xb5,0xd4]
     {vex} vpmadd52huq %xmm4, %xmm3, %xmm2

// CHECK: {vex} vpmadd52huq  268435456(%esp,%esi,8), %ymm3, %ymm2
// CHECK: encoding: [0xc4,0xe2,0xe5,0xb5,0x94,0xf4,0x00,0x00,0x00,0x10]
     {vex} vpmadd52huq  268435456(%esp,%esi,8), %ymm3, %ymm2

// CHECK: {vex} vpmadd52huq  291(%edi,%eax,4), %ymm3, %ymm2
// CHECK: encoding: [0xc4,0xe2,0xe5,0xb5,0x94,0x87,0x23,0x01,0x00,0x00]
     {vex} vpmadd52huq  291(%edi,%eax,4), %ymm3, %ymm2

// CHECK: {vex} vpmadd52huq  (%eax), %ymm3, %ymm2
// CHECK: encoding: [0xc4,0xe2,0xe5,0xb5,0x10]
     {vex} vpmadd52huq  (%eax), %ymm3, %ymm2

// CHECK: {vex} vpmadd52huq  -1024(,%ebp,2), %ymm3, %ymm2
// CHECK: encoding: [0xc4,0xe2,0xe5,0xb5,0x14,0x6d,0x00,0xfc,0xff,0xff]
     {vex} vpmadd52huq  -1024(,%ebp,2), %ymm3, %ymm2

// CHECK: {vex} vpmadd52huq  4064(%ecx), %ymm3, %ymm2
// CHECK: encoding: [0xc4,0xe2,0xe5,0xb5,0x91,0xe0,0x0f,0x00,0x00]
     {vex} vpmadd52huq  4064(%ecx), %ymm3, %ymm2

// CHECK: {vex} vpmadd52huq  -4096(%edx), %ymm3, %ymm2
// CHECK: encoding: [0xc4,0xe2,0xe5,0xb5,0x92,0x00,0xf0,0xff,0xff]
     {vex} vpmadd52huq  -4096(%edx), %ymm3, %ymm2

// CHECK: {vex} vpmadd52huq  268435456(%esp,%esi,8), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0xe1,0xb5,0x94,0xf4,0x00,0x00,0x00,0x10]
     {vex} vpmadd52huq  268435456(%esp,%esi,8), %xmm3, %xmm2

// CHECK: {vex} vpmadd52huq  291(%edi,%eax,4), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0xe1,0xb5,0x94,0x87,0x23,0x01,0x00,0x00]
     {vex} vpmadd52huq  291(%edi,%eax,4), %xmm3, %xmm2

// CHECK: {vex} vpmadd52huq  (%eax), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0xe1,0xb5,0x10]
     {vex} vpmadd52huq  (%eax), %xmm3, %xmm2

// CHECK: {vex} vpmadd52huq  -512(,%ebp,2), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0xe1,0xb5,0x14,0x6d,0x00,0xfe,0xff,0xff]
     {vex} vpmadd52huq  -512(,%ebp,2), %xmm3, %xmm2

// CHECK: {vex} vpmadd52huq  2032(%ecx), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0xe1,0xb5,0x91,0xf0,0x07,0x00,0x00]
     {vex} vpmadd52huq  2032(%ecx), %xmm3, %xmm2

// CHECK: {vex} vpmadd52huq  -2048(%edx), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0xe1,0xb5,0x92,0x00,0xf8,0xff,0xff]
     {vex} vpmadd52huq  -2048(%edx), %xmm3, %xmm2

// CHECK: {vex} vpmadd52luq %ymm4, %ymm3, %ymm2
// CHECK: encoding: [0xc4,0xe2,0xe5,0xb4,0xd4]
     {vex} vpmadd52luq %ymm4, %ymm3, %ymm2

// CHECK: {vex} vpmadd52luq %xmm4, %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0xe1,0xb4,0xd4]
     {vex} vpmadd52luq %xmm4, %xmm3, %xmm2

// CHECK: {vex} vpmadd52luq  268435456(%esp,%esi,8), %ymm3, %ymm2
// CHECK: encoding: [0xc4,0xe2,0xe5,0xb4,0x94,0xf4,0x00,0x00,0x00,0x10]
     {vex} vpmadd52luq  268435456(%esp,%esi,8), %ymm3, %ymm2

// CHECK: {vex} vpmadd52luq  291(%edi,%eax,4), %ymm3, %ymm2
// CHECK: encoding: [0xc4,0xe2,0xe5,0xb4,0x94,0x87,0x23,0x01,0x00,0x00]
     {vex} vpmadd52luq  291(%edi,%eax,4), %ymm3, %ymm2

// CHECK: {vex} vpmadd52luq  (%eax), %ymm3, %ymm2
// CHECK: encoding: [0xc4,0xe2,0xe5,0xb4,0x10]
     {vex} vpmadd52luq  (%eax), %ymm3, %ymm2

// CHECK: {vex} vpmadd52luq  -1024(,%ebp,2), %ymm3, %ymm2
// CHECK: encoding: [0xc4,0xe2,0xe5,0xb4,0x14,0x6d,0x00,0xfc,0xff,0xff]
     {vex} vpmadd52luq  -1024(,%ebp,2), %ymm3, %ymm2

// CHECK: {vex} vpmadd52luq  4064(%ecx), %ymm3, %ymm2
// CHECK: encoding: [0xc4,0xe2,0xe5,0xb4,0x91,0xe0,0x0f,0x00,0x00]
     {vex} vpmadd52luq  4064(%ecx), %ymm3, %ymm2

// CHECK: {vex} vpmadd52luq  -4096(%edx), %ymm3, %ymm2
// CHECK: encoding: [0xc4,0xe2,0xe5,0xb4,0x92,0x00,0xf0,0xff,0xff]
     {vex} vpmadd52luq  -4096(%edx), %ymm3, %ymm2

// CHECK: {vex} vpmadd52luq  268435456(%esp,%esi,8), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0xe1,0xb4,0x94,0xf4,0x00,0x00,0x00,0x10]
     {vex} vpmadd52luq  268435456(%esp,%esi,8), %xmm3, %xmm2

// CHECK: {vex} vpmadd52luq  291(%edi,%eax,4), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0xe1,0xb4,0x94,0x87,0x23,0x01,0x00,0x00]
     {vex} vpmadd52luq  291(%edi,%eax,4), %xmm3, %xmm2

// CHECK: {vex} vpmadd52luq  (%eax), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0xe1,0xb4,0x10]
     {vex} vpmadd52luq  (%eax), %xmm3, %xmm2

// CHECK: {vex} vpmadd52luq  -512(,%ebp,2), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0xe1,0xb4,0x14,0x6d,0x00,0xfe,0xff,0xff]
     {vex} vpmadd52luq  -512(,%ebp,2), %xmm3, %xmm2

// CHECK: {vex} vpmadd52luq  2032(%ecx), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0xe1,0xb4,0x91,0xf0,0x07,0x00,0x00]
     {vex} vpmadd52luq  2032(%ecx), %xmm3, %xmm2

// CHECK: {vex} vpmadd52luq  -2048(%edx), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0xe1,0xb4,0x92,0x00,0xf8,0xff,0xff]
     {vex} vpmadd52luq  -2048(%edx), %xmm3, %xmm2

