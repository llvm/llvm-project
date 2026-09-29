// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+pauth   < %s | FileCheck %s

paciasp

// CHECK: paciasp // encoding: [0x3f,0x23,0x03,0xd5]
