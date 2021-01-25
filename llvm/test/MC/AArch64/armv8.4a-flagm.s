// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+flagm   < %s | FileCheck %s

cfinv

// CHECK: .text
cfinv // encoding: [0x1f,0x40,0x00,0xd5]
