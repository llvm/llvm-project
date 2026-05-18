// RUN: llvm-mc --triple=aarch64-linux -mattr=+cmpbr --show-encoding --show-inst < %s | FileCheck %s
 .text

// CHECK: cbbge w5, w3, #-1024                  // encoding: [0x05,0xa0,0x23,0x74]
    cbble w3, w5, #-1024

// CHECK: cbbhi w5, w3, #-1024                  // encoding: [0x05,0xa0,0x43,0x74]
    cbblo w3, w5, #-1024

// CHECK: cbbhs w5, w3, #-1024                  // encoding: [0x05,0xa0,0x63,0x74]
    cbbls w3, w5, #-1024

// CHECK: cbbgt w5, w3, #-1024                  // encoding: [0x05,0xa0,0x03,0x74]
    cbblt w3, w5, #-1024

// CHECK: cbhge w5, w3, #-1024                  // encoding: [0x05,0xe0,0x23,0x74]
    cbhle w3, w5, #-1024

// CHECK: cbhhi w5, w3, #-1024                  // encoding: [0x05,0xe0,0x43,0x74]
    cbhlo w3, w5, #-1024

// CHECK: cbhhs w5, w3, #-1024                  // encoding: [0x05,0xe0,0x63,0x74]
    cbhls w3, w5, #-1024

// CHECK: cbhgt w5, w3, #-1024                  // encoding: [0x05,0xe0,0x03,0x74]
    cbhlt w3, w5, #-1024

// CHECK: cbgt w5, #0, #-1024                  // encoding: [0x05,0x20,0x00,0x75]
    cbge w5, #1, #-1024

// CHECK: cbgt x5, #63, #-1024                 // encoding: [0x05,0xa0,0x1f,0xf5]
    cbge x5, #64, #-1024

// CHECK: cbhi w5, #0, #-1024                  // encoding: [0x05,0x20,0x40,0x75]
    cbhs w5, #1, #-1024

// CHECK: cbhi x5, #63, #-1024                 // encoding: [0x05,0xa0,0x5f,0xf5]
    cbhs x5, #64, #-1024

// CHECK: cblt w5, #0, #-1024                  // encoding: [0x05,0x20,0x20,0x75]
    cble w5, #-1, #-1024

// CHECK: cblt x5, #63, #-1024                 // encoding: [0x05,0xa0,0x3f,0xf5]
    cble x5, #62, #-1024

// CHECK: cblo w5, #0, #-1024                  // encoding: [0x05,0x20,0x60,0x75]
    cbls w5, #-1, #-1024

// CHECK: cblo x5, #63, #-1024                 // encoding: [0x05,0xa0,0x7f,0xf5]
    cbls x5, #62, #-1024

// CHECK: cbge w5, w3, #-1024                  // encoding: [0x05,0x20,0x23,0x74]
    cble w3, w5, #-1024

// CHECK: cbge x5, x3, #-1024                  // encoding: [0x05,0x20,0x23,0xf4]
    cble x3, x5, #-1024

// CHECK: cbhi w5, w3, #-1024                  // encoding: [0x05,0x20,0x43,0x74]
    cblo w3, w5, #-1024

// CHECK: cbhi x5, x3, #-1024                  // encoding: [0x05,0x20,0x43,0xf4]
    cblo x3, x5, #-1024

// CHECK: cbhs w5, w3, #-1024                  // encoding: [0x05,0x20,0x63,0x74]
    cbls w3, w5, #-1024

// CHECK: cbhs x5, x3, #-1024                  // encoding: [0x05,0x20,0x63,0xf4]
    cbls x3, x5, #-1024

// CHECK: cbgt w5, w3, #-1024                  // encoding: [0x05,0x20,0x03,0x74]
    cblt w3, w5, #-1024

// CHECK: cbgt x5, x3, #-1024                  // encoding: [0x05,0x20,0x03,0xf4]
    cblt x3, x5, #-1024
