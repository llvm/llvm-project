// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+wfxt < %s | FileCheck %s
// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.7a < %s | FileCheck %s
// RUN: not llvm-mc -triple aarch64-none-linux-gnu < %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-NO-WFxT-ERR %s < %t

  wfet x17
// CHECK: wfet x17                      // encoding: [0x11,0x10,0x03,0xd5]
// CHECK-NO-WFxT-ERR: [[@LINE-2]]:3: error: instruction requires: wfxt

  wfit x3
// CHECK: wfit x3                       // encoding: [0x23,0x10,0x03,0xd5]
// CHECK-NO-WFxT-ERR: [[@LINE-2]]:3: error: instruction requires: wfxt
