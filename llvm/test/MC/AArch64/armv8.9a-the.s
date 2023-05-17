// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding               -mattr=+the -mattr=+d128 < %s | FileCheck %s
// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.9a -mattr=+the -mattr=+d128 < %s | FileCheck %s
// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v9.4a -mattr=+the -mattr=+d128 < %s | FileCheck %s

// RUN: not llvm-mc -triple aarch64-none-linux-gnu                           < %s 2>&1 | FileCheck --check-prefix=ERROR-NO-THE %s
// RUN: not llvm-mc -triple aarch64-none-linux-gnu -mattr=+v8.9a             < %s 2>&1 | FileCheck --check-prefix=ERROR-NO-THE %s
// RUN: not llvm-mc -triple aarch64-none-linux-gnu -mattr=+v9.4a             < %s 2>&1 | FileCheck --check-prefix=ERROR-NO-THE %s

// RUN: not llvm-mc -triple aarch64-none-linux-gnu               -mattr=+the < %s 2>&1 | FileCheck --check-prefix=ERROR-NO-D128 %s
// RUN: not llvm-mc -triple aarch64-none-linux-gnu -mattr=+v8.9a -mattr=+the < %s 2>&1 | FileCheck --check-prefix=ERROR-NO-D128 %s
// RUN: not llvm-mc -triple aarch64-none-linux-gnu -mattr=+v9.4a -mattr=+the < %s 2>&1 | FileCheck --check-prefix=ERROR-NO-D128 %s

// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+the -mattr=+d128 < %s 2>&1 | FileCheck --check-prefix=ERROR-NO-ZXR %s

            mrs x3, RCWMASK_EL1
// CHECK:   mrs x3, RCWMASK_EL1   // encoding: [0xc3,0xd0,0x38,0xd5]
// ERROR-NO-THE: [[@LINE-2]]:21: error: expected readable system register
            msr RCWMASK_EL1, x1
// CHECK:   msr RCWMASK_EL1, x1   // encoding: [0xc1,0xd0,0x18,0xd5]
// ERROR-NO-THE: [[@LINE-2]]:17: error: expected writable system register or pstate
            mrs x3, RCWSMASK_EL1
// CHECK:   mrs x3, RCWSMASK_EL1  // encoding: [0x63,0xd0,0x38,0xd5]
// ERROR-NO-THE: [[@LINE-2]]:21: error: expected readable system register
            msr RCWSMASK_EL1, x1
// CHECK:   msr RCWSMASK_EL1, x1  // encoding: [0x61,0xd0,0x18,0xd5]
// ERROR-NO-THE: [[@LINE-2]]:17: error: expected writable system register or pstate

            rcwcas   x0, x1, [x4]
// CHECK:   rcwcas   x0, x1, [x4] // encoding: [0x81,0x08,0x20,0x19]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the
            rcwcasa  x0, x1, [x4]
// CHECK:   rcwcasa  x0, x1, [x4] // encoding: [0x81,0x08,0xa0,0x19]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the
            rcwcasal x0, x1, [x4]
// CHECK:   rcwcasal x0, x1, [x4] // encoding: [0x81,0x08,0xe0,0x19]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the
            rcwcasl  x0, x1, [x4]
// CHECK:   rcwcasl  x0, x1, [x4] // encoding: [0x81,0x08,0x60,0x19]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the
            rcwcas   x3, x5, [sp]
// CHECK:   rcwcas   x3, x5, [sp] // encoding: [0xe5,0x0b,0x23,0x19]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the
            rcwcasa  x3, x5, [sp]
// CHECK:   rcwcasa  x3, x5, [sp] // encoding: [0xe5,0x0b,0xa3,0x19]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the
            rcwcasal x3, x5, [sp]
// CHECK:   rcwcasal x3, x5, [sp] // encoding: [0xe5,0x0b,0xe3,0x19]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the
            rcwcasl  x3, x5, [sp]
// CHECK:   rcwcasl  x3, x5, [sp] // encoding: [0xe5,0x0b,0x63,0x19]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the

            rcwscas   x0, x1, [x4]
// CHECK:   rcwscas   x0, x1, [x4] // encoding: [0x81,0x08,0x20,0x59]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the
            rcwscasa  x0, x1, [x4]
// CHECK:   rcwscasa  x0, x1, [x4] // encoding: [0x81,0x08,0xa0,0x59]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the
            rcwscasal x0, x1, [x4]
// CHECK:   rcwscasal x0, x1, [x4] // encoding: [0x81,0x08,0xe0,0x59]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the
            rcwscasl  x0, x1, [x4]
// CHECK:   rcwscasl  x0, x1, [x4] // encoding: [0x81,0x08,0x60,0x59]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the
            rcwscas   x3, x5, [sp]
// CHECK:   rcwscas   x3, x5, [sp] // encoding: [0xe5,0x0b,0x23,0x59]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the
            rcwscasa  x3, x5, [sp]
// CHECK:   rcwscasa  x3, x5, [sp] // encoding: [0xe5,0x0b,0xa3,0x59]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the
            rcwscasal x3, x5, [sp]
// CHECK:   rcwscasal x3, x5, [sp] // encoding: [0xe5,0x0b,0xe3,0x59]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the
            rcwscasl  x3, x5, [sp]
// CHECK:   rcwscasl  x3, x5, [sp] // encoding: [0xe5,0x0b,0x63,0x59]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the

            rcwcasp   x0, x1, x6, x7, [x4]
// CHECK:   rcwcasp   x0, x1, x6, x7, [x4] // encoding: [0x86,0x0c,0x20,0x19]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128
            rcwcaspa  x0, x1, x6, x7, [x4]
// CHECK:   rcwcaspa  x0, x1, x6, x7, [x4] // encoding: [0x86,0x0c,0xa0,0x19]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128
            rcwcaspal x0, x1, x6, x7, [x4]
// CHECK:   rcwcaspal x0, x1, x6, x7, [x4] // encoding: [0x86,0x0c,0xe0,0x19]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128
            rcwcaspl  x0, x1, x6, x7, [x4]
// CHECK:   rcwcaspl  x0, x1, x6, x7, [x4] // encoding: [0x86,0x0c,0x60,0x19]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128
            rcwcasp   x4, x5, x6, x7, [sp]
// CHECK:   rcwcasp   x4, x5, x6, x7, [sp] // encoding: [0xe6,0x0f,0x24,0x19]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128
            rcwcaspa  x4, x5, x6, x7, [sp]
// CHECK:   rcwcaspa  x4, x5, x6, x7, [sp] // encoding: [0xe6,0x0f,0xa4,0x19]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128
            rcwcaspal x4, x5, x6, x7, [sp]
// CHECK:   rcwcaspal x4, x5, x6, x7, [sp] // encoding: [0xe6,0x0f,0xe4,0x19]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128
            rcwcaspl  x4, x5, x6, x7, [sp]
// CHECK:   rcwcaspl  x4, x5, x6, x7, [sp] // encoding: [0xe6,0x0f,0x64,0x19]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128

            rcwscasp   x0, x1, x6, x7, [x4]
// CHECK:   rcwscasp   x0, x1, x6, x7, [x4] // encoding: [0x86,0x0c,0x20,0x59]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128
            rcwscaspa  x0, x1, x6, x7, [x4]
// CHECK:   rcwscaspa  x0, x1, x6, x7, [x4] // encoding: [0x86,0x0c,0xa0,0x59]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128
            rcwscaspal x0, x1, x6, x7, [x4]
// CHECK:   rcwscaspal x0, x1, x6, x7, [x4] // encoding: [0x86,0x0c,0xe0,0x59]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128
            rcwscaspl  x0, x1, x6, x7, [x4]
// CHECK:   rcwscaspl  x0, x1, x6, x7, [x4] // encoding: [0x86,0x0c,0x60,0x59]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128
            rcwscasp   x4, x5, x6, x7, [sp]
// CHECK:   rcwscasp   x4, x5, x6, x7, [sp] // encoding: [0xe6,0x0f,0x24,0x59]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128
            rcwscaspa  x4, x5, x6, x7, [sp]
// CHECK:   rcwscaspa  x4, x5, x6, x7, [sp] // encoding: [0xe6,0x0f,0xa4,0x59]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128
            rcwscaspal x4, x5, x6, x7, [sp]
// CHECK:   rcwscaspal x4, x5, x6, x7, [sp] // encoding: [0xe6,0x0f,0xe4,0x59]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128
            rcwscaspl  x4, x5, x6, x7, [sp]
// CHECK:   rcwscaspl  x4, x5, x6, x7, [sp] // encoding: [0xe6,0x0f,0x64,0x59]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128

            rcwclr   x0, x1, [x4]
// CHECK:   rcwclr   x0, x1, [x4] // encoding: [0x81,0x90,0x20,0x38]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the
            rcwclra  x0, x1, [x4]
// CHECK:   rcwclra  x0, x1, [x4] // encoding: [0x81,0x90,0xa0,0x38]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the
            rcwclral x0, x1, [x4]
// CHECK:   rcwclral x0, x1, [x4] // encoding: [0x81,0x90,0xe0,0x38]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the
            rcwclrl  x0, x1, [x4]
// CHECK:   rcwclrl  x0, x1, [x4] // encoding: [0x81,0x90,0x60,0x38]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the
            rcwclr   x3, x5, [sp]
// CHECK:   rcwclr   x3, x5, [sp] // encoding: [0xe5,0x93,0x23,0x38]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the
            rcwclra  x3, x5, [sp]
// CHECK:   rcwclra  x3, x5, [sp] // encoding: [0xe5,0x93,0xa3,0x38]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the
            rcwclral x3, x5, [sp]
// CHECK:   rcwclral x3, x5, [sp] // encoding: [0xe5,0x93,0xe3,0x38]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the
            rcwclrl  x3, x5, [sp]
// CHECK:   rcwclrl  x3, x5, [sp] // encoding: [0xe5,0x93,0x63,0x38]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the

            rcwsclr   x0, x1, [x4]
// CHECK:   rcwsclr   x0, x1, [x4] // encoding: [0x81,0x90,0x20,0x78]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the
            rcwsclra  x0, x1, [x4]
// CHECK:   rcwsclra  x0, x1, [x4] // encoding: [0x81,0x90,0xa0,0x78]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the
            rcwsclral x0, x1, [x4]
// CHECK:   rcwsclral x0, x1, [x4] // encoding: [0x81,0x90,0xe0,0x78]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the
            rcwsclrl  x0, x1, [x4]
// CHECK:   rcwsclrl  x0, x1, [x4] // encoding: [0x81,0x90,0x60,0x78]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the
            rcwsclr   x3, x5, [sp]
// CHECK:   rcwsclr   x3, x5, [sp] // encoding: [0xe5,0x93,0x23,0x78]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the
            rcwsclra  x3, x5, [sp]
// CHECK:   rcwsclra  x3, x5, [sp] // encoding: [0xe5,0x93,0xa3,0x78]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the
            rcwsclral x3, x5, [sp]
// CHECK:   rcwsclral x3, x5, [sp] // encoding: [0xe5,0x93,0xe3,0x78]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the
            rcwsclrl  x3, x5, [sp]
// CHECK:   rcwsclrl  x3, x5, [sp] // encoding: [0xe5,0x93,0x63,0x78]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the

            rcwclrp   x1, x0, [x4]
// CHECK:   rcwclrp   x1, x0, [x4] // encoding: [0x81,0x90,0x20,0x19]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128
            rcwclrpa  x1, x0, [x4]
// CHECK:   rcwclrpa  x1, x0, [x4] // encoding: [0x81,0x90,0xa0,0x19]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128
            rcwclrpal x1, x0, [x4]
// CHECK:   rcwclrpal x1, x0, [x4] // encoding: [0x81,0x90,0xe0,0x19]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128
            rcwclrpl  x1, x0, [x4]
// CHECK:   rcwclrpl  x1, x0, [x4] // encoding: [0x81,0x90,0x60,0x19]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128
            rcwclrp   x5, x3, [sp]
// CHECK:   rcwclrp   x5, x3, [sp] // encoding: [0xe5,0x93,0x23,0x19]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128
            rcwclrpa  x5, x3, [sp]
// CHECK:   rcwclrpa  x5, x3, [sp] // encoding: [0xe5,0x93,0xa3,0x19]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128
            rcwclrpal x5, x3, [sp]
// CHECK:   rcwclrpal x5, x3, [sp] // encoding: [0xe5,0x93,0xe3,0x19]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128
            rcwclrpl  x5, x3, [sp]
// CHECK:   rcwclrpl  x5, x3, [sp] // encoding: [0xe5,0x93,0x63,0x19]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128

            rcwsclrp   x1, x0, [x4]
// CHECK:   rcwsclrp   x1, x0, [x4] // encoding: [0x81,0x90,0x20,0x59]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128
            rcwsclrpa  x1, x0, [x4]
// CHECK:   rcwsclrpa  x1, x0, [x4] // encoding: [0x81,0x90,0xa0,0x59]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128
            rcwsclrpal x1, x0, [x4]
// CHECK:   rcwsclrpal x1, x0, [x4] // encoding: [0x81,0x90,0xe0,0x59]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128
            rcwsclrpl  x1, x0, [x4]
// CHECK:   rcwsclrpl  x1, x0, [x4] // encoding: [0x81,0x90,0x60,0x59]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128
            rcwsclrp   x5, x3, [sp]
// CHECK:   rcwsclrp   x5, x3, [sp] // encoding: [0xe5,0x93,0x23,0x59]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128
            rcwsclrpa  x5, x3, [sp]
// CHECK:   rcwsclrpa  x5, x3, [sp] // encoding: [0xe5,0x93,0xa3,0x59]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128
            rcwsclrpal x5, x3, [sp]
// CHECK:   rcwsclrpal x5, x3, [sp] // encoding: [0xe5,0x93,0xe3,0x59]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128
            rcwsclrpl  x5, x3, [sp]
// CHECK:   rcwsclrpl  x5, x3, [sp] // encoding: [0xe5,0x93,0x63,0x59]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128

            rcwset   x0, x1, [x4]
// CHECK:   rcwset   x0, x1, [x4] // encoding: [0x81,0xb0,0x20,0x38]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the
            rcwseta  x0, x1, [x4]
// CHECK:   rcwseta  x0, x1, [x4] // encoding: [0x81,0xb0,0xa0,0x38]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the
            rcwsetal x0, x1, [x4]
// CHECK:   rcwsetal x0, x1, [x4] // encoding: [0x81,0xb0,0xe0,0x38]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the
            rcwsetl  x0, x1, [x4]
// CHECK:   rcwsetl  x0, x1, [x4] // encoding: [0x81,0xb0,0x60,0x38]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the
            rcwset   x3, x5, [sp]
// CHECK:   rcwset   x3, x5, [sp] // encoding: [0xe5,0xb3,0x23,0x38]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the
            rcwseta  x3, x5, [sp]
// CHECK:   rcwseta  x3, x5, [sp] // encoding: [0xe5,0xb3,0xa3,0x38]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the
            rcwsetal x3, x5, [sp]
// CHECK:   rcwsetal x3, x5, [sp] // encoding: [0xe5,0xb3,0xe3,0x38]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the
            rcwsetl  x3, x5, [sp]
// CHECK:   rcwsetl  x3, x5, [sp] // encoding: [0xe5,0xb3,0x63,0x38]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the

            rcwsset   x0, x1, [x4]
// CHECK:   rcwsset   x0, x1, [x4] // encoding: [0x81,0xb0,0x20,0x78]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the
            rcwsseta  x0, x1, [x4]
// CHECK:   rcwsseta  x0, x1, [x4] // encoding: [0x81,0xb0,0xa0,0x78]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the
            rcwssetal x0, x1, [x4]
// CHECK:   rcwssetal x0, x1, [x4] // encoding: [0x81,0xb0,0xe0,0x78]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the
            rcwssetl  x0, x1, [x4]
// CHECK:   rcwssetl  x0, x1, [x4] // encoding: [0x81,0xb0,0x60,0x78]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the
            rcwsset   x3, x5, [sp]
// CHECK:   rcwsset   x3, x5, [sp] // encoding: [0xe5,0xb3,0x23,0x78]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the
            rcwsseta  x3, x5, [sp]
// CHECK:   rcwsseta  x3, x5, [sp] // encoding: [0xe5,0xb3,0xa3,0x78]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the
            rcwssetal x3, x5, [sp]
// CHECK:   rcwssetal x3, x5, [sp] // encoding: [0xe5,0xb3,0xe3,0x78]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the
            rcwssetl  x3, x5, [sp]
// CHECK:   rcwssetl  x3, x5, [sp] // encoding: [0xe5,0xb3,0x63,0x78]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the

            rcwsetp   x1, x0, [x4]
// CHECK:   rcwsetp   x1, x0, [x4] // encoding: [0x81,0xb0,0x20,0x19]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128
            rcwsetpa  x1, x0, [x4]
// CHECK:   rcwsetpa  x1, x0, [x4] // encoding: [0x81,0xb0,0xa0,0x19]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128
            rcwsetpal x1, x0, [x4]
// CHECK:   rcwsetpal x1, x0, [x4] // encoding: [0x81,0xb0,0xe0,0x19]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128
            rcwsetpl  x1, x0, [x4]
// CHECK:   rcwsetpl  x1, x0, [x4] // encoding: [0x81,0xb0,0x60,0x19]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128
            rcwsetp   x5, x3, [sp]
// CHECK:   rcwsetp   x5, x3, [sp] // encoding: [0xe5,0xb3,0x23,0x19]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128
            rcwsetpa  x5, x3, [sp]
// CHECK:   rcwsetpa  x5, x3, [sp] // encoding: [0xe5,0xb3,0xa3,0x19]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128
            rcwsetpal x5, x3, [sp]
// CHECK:   rcwsetpal x5, x3, [sp] // encoding: [0xe5,0xb3,0xe3,0x19]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128
            rcwsetpl  x5, x3, [sp]
// CHECK:   rcwsetpl  x5, x3, [sp] // encoding: [0xe5,0xb3,0x63,0x19]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128

            rcwssetp   x1, x0, [x4]
// CHECK:   rcwssetp   x1, x0, [x4] // encoding: [0x81,0xb0,0x20,0x59]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128
            rcwssetpa  x1, x0, [x4]
// CHECK:   rcwssetpa  x1, x0, [x4] // encoding: [0x81,0xb0,0xa0,0x59]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128
            rcwssetpal x1, x0, [x4]
// CHECK:   rcwssetpal x1, x0, [x4] // encoding: [0x81,0xb0,0xe0,0x59]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128
            rcwssetpl  x1, x0, [x4]
// CHECK:   rcwssetpl  x1, x0, [x4] // encoding: [0x81,0xb0,0x60,0x59]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128
            rcwssetp   x5, x3, [sp]
// CHECK:   rcwssetp   x5, x3, [sp] // encoding: [0xe5,0xb3,0x23,0x59]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128
            rcwssetpa  x5, x3, [sp]
// CHECK:   rcwssetpa  x5, x3, [sp] // encoding: [0xe5,0xb3,0xa3,0x59]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128
            rcwssetpal x5, x3, [sp]
// CHECK:   rcwssetpal x5, x3, [sp] // encoding: [0xe5,0xb3,0xe3,0x59]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128
            rcwssetpl  x5, x3, [sp]
// CHECK:   rcwssetpl  x5, x3, [sp] // encoding: [0xe5,0xb3,0x63,0x59]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128

            rcwswp   x0, x1, [x4]
// CHECK:   rcwswp   x0, x1, [x4] // encoding: [0x81,0xa0,0x20,0x38]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the
            rcwswpa  x0, x1, [x4]
// CHECK:   rcwswpa  x0, x1, [x4] // encoding: [0x81,0xa0,0xa0,0x38]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the
            rcwswpal x0, x1, [x4]
// CHECK:   rcwswpal x0, x1, [x4] // encoding: [0x81,0xa0,0xe0,0x38]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the
            rcwswpl  x0, x1, [x4]
// CHECK:   rcwswpl  x0, x1, [x4] // encoding: [0x81,0xa0,0x60,0x38]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the
            rcwswp   x3, x5, [sp]
// CHECK:   rcwswp   x3, x5, [sp] // encoding: [0xe5,0xa3,0x23,0x38]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the
            rcwswpa  x3, x5, [sp]
// CHECK:   rcwswpa  x3, x5, [sp] // encoding: [0xe5,0xa3,0xa3,0x38]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the
            rcwswpal x3, x5, [sp]
// CHECK:   rcwswpal x3, x5, [sp] // encoding: [0xe5,0xa3,0xe3,0x38]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the
            rcwswpl  x3, x5, [sp]
// CHECK:   rcwswpl  x3, x5, [sp] // encoding: [0xe5,0xa3,0x63,0x38]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the

            rcwsswp   x0, x1, [x4]
// CHECK:   rcwsswp   x0, x1, [x4] // encoding: [0x81,0xa0,0x20,0x78]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the
            rcwsswpa  x0, x1, [x4]
// CHECK:   rcwsswpa  x0, x1, [x4] // encoding: [0x81,0xa0,0xa0,0x78]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the
            rcwsswpal x0, x1, [x4]
// CHECK:   rcwsswpal x0, x1, [x4] // encoding: [0x81,0xa0,0xe0,0x78]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the
            rcwsswpl  x0, x1, [x4]
// CHECK:   rcwsswpl  x0, x1, [x4] // encoding: [0x81,0xa0,0x60,0x78]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the
            rcwsswp   x3, x5, [sp]
// CHECK:   rcwsswp   x3, x5, [sp] // encoding: [0xe5,0xa3,0x23,0x78]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the
            rcwsswpa  x3, x5, [sp]
// CHECK:   rcwsswpa  x3, x5, [sp] // encoding: [0xe5,0xa3,0xa3,0x78]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the
            rcwsswpal x3, x5, [sp]
// CHECK:   rcwsswpal x3, x5, [sp] // encoding: [0xe5,0xa3,0xe3,0x78]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the
            rcwsswpl  x3, x5, [sp]
// CHECK:   rcwsswpl  x3, x5, [sp] // encoding: [0xe5,0xa3,0x63,0x78]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: the

            rcwswpp   x1, x0, [x4]
// CHECK:   rcwswpp   x1, x0, [x4] // encoding: [0x81,0xa0,0x20,0x19]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128
            rcwswppa  x1, x0, [x4]
// CHECK:   rcwswppa  x1, x0, [x4] // encoding: [0x81,0xa0,0xa0,0x19]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128
            rcwswppal x1, x0, [x4]
// CHECK:   rcwswppal x1, x0, [x4] // encoding: [0x81,0xa0,0xe0,0x19]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128
            rcwswppl  x1, x0, [x4]
// CHECK:   rcwswppl  x1, x0, [x4] // encoding: [0x81,0xa0,0x60,0x19]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128
            rcwswpp   x5, x3, [sp]
// CHECK:   rcwswpp   x5, x3, [sp] // encoding: [0xe5,0xa3,0x23,0x19]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128
            rcwswppa  x5, x3, [sp]
// CHECK:   rcwswppa  x5, x3, [sp] // encoding: [0xe5,0xa3,0xa3,0x19]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128
            rcwswppal x5, x3, [sp]
// CHECK:   rcwswppal x5, x3, [sp] // encoding: [0xe5,0xa3,0xe3,0x19]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128
            rcwswppl  x5, x3, [sp]
// CHECK:   rcwswppl  x5, x3, [sp] // encoding: [0xe5,0xa3,0x63,0x19]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128

            rcwsswpp   x1, x0, [x4]
// CHECK:   rcwsswpp   x1, x0, [x4] // encoding: [0x81,0xa0,0x20,0x59]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128
            rcwsswppa  x1, x0, [x4]
// CHECK:   rcwsswppa  x1, x0, [x4] // encoding: [0x81,0xa0,0xa0,0x59]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128
            rcwsswppal x1, x0, [x4]
// CHECK:   rcwsswppal x1, x0, [x4] // encoding: [0x81,0xa0,0xe0,0x59]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128
            rcwsswppl  x1, x0, [x4]
// CHECK:   rcwsswppl  x1, x0, [x4] // encoding: [0x81,0xa0,0x60,0x59]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128
            rcwsswpp   x5, x3, [sp]
// CHECK:   rcwsswpp   x5, x3, [sp] // encoding: [0xe5,0xa3,0x23,0x59]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128
            rcwsswppa  x5, x3, [sp]
// CHECK:   rcwsswppa  x5, x3, [sp] // encoding: [0xe5,0xa3,0xa3,0x59]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128
            rcwsswppal x5, x3, [sp]
// CHECK:   rcwsswppal x5, x3, [sp] // encoding: [0xe5,0xa3,0xe3,0x59]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128
            rcwsswppl  x5, x3, [sp]
// CHECK:   rcwsswppl  x5, x3, [sp] // encoding: [0xe5,0xa3,0x63,0x59]
// ERROR-NO-THE: [[@LINE-2]]:13: error: instruction requires: d128 the
// ERROR-NO-D128: [[@LINE-3]]:13: error: instruction requires: d128

            rcwswpp   xzr, x5, [x4]
// ERROR-NO-ZXR:   [[@LINE-1]]:23: error: invalid operand for instruction
            rcwswppa  xzr, x5, [x4]
// ERROR-NO-ZXR:   [[@LINE-1]]:23: error: invalid operand for instruction
            rcwswppal xzr, x5, [x4]
// ERROR-NO-ZXR:   [[@LINE-1]]:23: error: invalid operand for instruction
            rcwswppl  xzr, x5, [x4]
// ERROR-NO-ZXR:   [[@LINE-1]]:23: error: invalid operand for instruction
            rcwswpp   x5, xzr, [x4]
// ERROR-NO-ZXR:   [[@LINE-1]]:27: error: invalid operand for instruction
            rcwswppa  x5, xzr, [x4]
// ERROR-NO-ZXR:   [[@LINE-1]]:27: error: invalid operand for instruction
            rcwswppal x5, xzr, [x4]
// ERROR-NO-ZXR:   [[@LINE-1]]:27: error: invalid operand for instruction
            rcwswppl  x5, xzr, [x4]
// ERROR-NO-ZXR:   [[@LINE-1]]:27: error: invalid operand for instruction

            rcwclrp   xzr, x5, [x4]
// ERROR-NO-ZXR:   [[@LINE-1]]:23: error: invalid operand for instruction
            rcwclrpa  xzr, x5, [x4]
// ERROR-NO-ZXR:   [[@LINE-1]]:23: error: invalid operand for instruction
            rcwclrpal xzr, x5, [x4]
// ERROR-NO-ZXR:   [[@LINE-1]]:23: error: invalid operand for instruction
            rcwclrpl  xzr, x5, [x4]
// ERROR-NO-ZXR:   [[@LINE-1]]:23: error: invalid operand for instruction
            rcwclrp   x5, xzr, [x4]
// ERROR-NO-ZXR:   [[@LINE-1]]:27: error: invalid operand for instruction
            rcwclrpa  x5, xzr, [x4]
// ERROR-NO-ZXR:   [[@LINE-1]]:27: error: invalid operand for instruction
            rcwclrpal x5, xzr, [x4]
// ERROR-NO-ZXR:   [[@LINE-1]]:27: error: invalid operand for instruction
            rcwclrpl  x5, xzr, [x4]
// ERROR-NO-ZXR:   [[@LINE-1]]:27: error: invalid operand for instruction

            rcwsetp   xzr, x5, [x4]
// ERROR-NO-ZXR:   [[@LINE-1]]:23: error: invalid operand for instruction
            rcwsetpa  xzr, x5, [x4]
// ERROR-NO-ZXR:   [[@LINE-1]]:23: error: invalid operand for instruction
            rcwsetpal xzr, x5, [x4]
// ERROR-NO-ZXR:   [[@LINE-1]]:23: error: invalid operand for instruction
            rcwsetpl  xzr, x5, [x4]
// ERROR-NO-ZXR:   [[@LINE-1]]:23: error: invalid operand for instruction
            rcwsetp   x5, xzr, [x4]
// ERROR-NO-ZXR:   [[@LINE-1]]:27: error: invalid operand for instruction
            rcwsetpa  x5, xzr, [x4]
// ERROR-NO-ZXR:   [[@LINE-1]]:27: error: invalid operand for instruction
            rcwsetpal x5, xzr, [x4]
// ERROR-NO-ZXR:   [[@LINE-1]]:27: error: invalid operand for instruction
            rcwsetpl  x5, xzr, [x4]
// ERROR-NO-ZXR:   [[@LINE-1]]:27: error: invalid operand for instruction

            rcwsswpp   xzr, x5, [x4]
// ERROR-NO-ZXR:   [[@LINE-1]]:24: error: invalid operand for instruction
            rcwsswppa  xzr, x5, [x4]
// ERROR-NO-ZXR:   [[@LINE-1]]:24: error: invalid operand for instruction
            rcwsswppal xzr, x5, [x4]
// ERROR-NO-ZXR:   [[@LINE-1]]:24: error: invalid operand for instruction
            rcwsswppl  xzr, x5, [x4]
// ERROR-NO-ZXR:   [[@LINE-1]]:24: error: invalid operand for instruction
            rcwsswpp   x5, xzr, [x4]
// ERROR-NO-ZXR:   [[@LINE-1]]:28: error: invalid operand for instruction
            rcwsswppa  x5, xzr, [x4]
// ERROR-NO-ZXR:   [[@LINE-1]]:28: error: invalid operand for instruction
            rcwsswppal x5, xzr, [x4]
// ERROR-NO-ZXR:   [[@LINE-1]]:28: error: invalid operand for instruction
            rcwsswppl  x5, xzr, [x4]
// ERROR-NO-ZXR:   [[@LINE-1]]:28: error: invalid operand for instruction

            rcwsclrp   xzr, x5, [x4]
// ERROR-NO-ZXR:   [[@LINE-1]]:24: error: invalid operand for instruction
            rcwsclrpa  xzr, x5, [x4]
// ERROR-NO-ZXR:   [[@LINE-1]]:24: error: invalid operand for instruction
            rcwsclrpal xzr, x5, [x4]
// ERROR-NO-ZXR:   [[@LINE-1]]:24: error: invalid operand for instruction
            rcwsclrpl  xzr, x5, [x4]
// ERROR-NO-ZXR:   [[@LINE-1]]:24: error: invalid operand for instruction
            rcwsclrp   x5, xzr, [x4]
// ERROR-NO-ZXR:   [[@LINE-1]]:28: error: invalid operand for instruction
            rcwsclrpa  x5, xzr, [x4]
// ERROR-NO-ZXR:   [[@LINE-1]]:28: error: invalid operand for instruction
            rcwsclrpal x5, xzr, [x4]
// ERROR-NO-ZXR:   [[@LINE-1]]:28: error: invalid operand for instruction
            rcwsclrpl  x5, xzr, [x4]
// ERROR-NO-ZXR:   [[@LINE-1]]:28: error: invalid operand for instruction

            rcwssetp   xzr, x5, [x4]
// ERROR-NO-ZXR:   [[@LINE-1]]:24: error: invalid operand for instruction
            rcwssetpa  xzr, x5, [x4]
// ERROR-NO-ZXR:   [[@LINE-1]]:24: error: invalid operand for instruction
            rcwssetpal xzr, x5, [x4]
// ERROR-NO-ZXR:   [[@LINE-1]]:24: error: invalid operand for instruction
            rcwssetpl  xzr, x5, [x4]
// ERROR-NO-ZXR:   [[@LINE-1]]:24: error: invalid operand for instruction
            rcwssetp   x5, xzr, [x4]
// ERROR-NO-ZXR:   [[@LINE-1]]:28: error: invalid operand for instruction
            rcwssetpa  x5, xzr, [x4]
// ERROR-NO-ZXR:   [[@LINE-1]]:28: error: invalid operand for instruction
            rcwssetpal x5, xzr, [x4]
// ERROR-NO-ZXR:   [[@LINE-1]]:28: error: invalid operand for instruction
            rcwssetpl  x5, xzr, [x4]
// ERROR-NO-ZXR:   [[@LINE-1]]:28: error: invalid operand for instruction
