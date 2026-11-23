// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+the,+d128 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+the,+d128,v8.9a < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+the,+d128,v9.4a < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+the,+d128 < %s \
// RUN:        | llvm-objdump -d --mattr=+the,+d128 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+the,+d128 < %s \
// RUN:   | llvm-objdump -d --mattr=-the,-d128 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+the,+d128 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+the,+d128 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -mattr=+the < %s 2>&1 | FileCheck --check-prefix=ERROR-NO-D128 %s


mrs x3, RCWMASK_EL1
// CHECK-INST: mrs x3, RCWMASK_EL1
// CHECK-ENCODING: encoding: [0xc3,0xd0,0x38,0xd5]
// CHECK-ERROR: error: expected readable system register
// CHECK-UNKNOWN:  d538d0c3      mrs x3, S3_0_C13_C0_6

msr RCWMASK_EL1, x1
// CHECK-INST: msr RCWMASK_EL1, x1
// CHECK-ENCODING: encoding: [0xc1,0xd0,0x18,0xd5]
// CHECK-ERROR: error: expected writable system register or pstate
// CHECK-UNKNOWN:  d518d0c1      msr S3_0_C13_C0_6, x1

mrs x3, RCWSMASK_EL1
// CHECK-INST: mrs x3, RCWSMASK_EL1
// CHECK-ENCODING: encoding: [0x63,0xd0,0x38,0xd5]
// CHECK-ERROR: error: expected readable system register
// CHECK-UNKNOWN:  d538d063      mrs x3, S3_0_C13_C0_3

msr RCWSMASK_EL1, x1
// CHECK-INST: msr RCWSMASK_EL1, x1
// CHECK-ENCODING: encoding: [0x61,0xd0,0x18,0xd5]
// CHECK-ERROR: error: expected writable system register or pstate
// CHECK-UNKNOWN:  d518d061      msr S3_0_C13_C0_3, x1

rcwcas   x0, x1, [x4]
// CHECK-INST: rcwcas x0, x1, [x4]
// CHECK-ENCODING: encoding: [0x81,0x08,0x20,0x19]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  19200881      <unknown>

rcwcasa  x0, x1, [x4]
// CHECK-INST: rcwcasa x0, x1, [x4]
// CHECK-ENCODING: encoding: [0x81,0x08,0xa0,0x19]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  19a00881      <unknown>

rcwcasal x0, x1, [x4]
// CHECK-INST: rcwcasal x0, x1, [x4]
// CHECK-ENCODING: encoding: [0x81,0x08,0xe0,0x19]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  19e00881      <unknown>

rcwcasl  x0, x1, [x4]
// CHECK-INST: rcwcasl x0, x1, [x4]
// CHECK-ENCODING: encoding: [0x81,0x08,0x60,0x19]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  19600881      <unknown>

rcwcas   x3, x5, [sp]
// CHECK-INST: rcwcas x3, x5, [sp]
// CHECK-ENCODING: encoding: [0xe5,0x0b,0x23,0x19]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  19230be5      <unknown>

rcwcasa  x3, x5, [sp]
// CHECK-INST: rcwcasa x3, x5, [sp]
// CHECK-ENCODING: encoding: [0xe5,0x0b,0xa3,0x19]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  19a30be5      <unknown>

rcwcasal x3, x5, [sp]
// CHECK-INST: rcwcasal x3, x5, [sp]
// CHECK-ENCODING: encoding: [0xe5,0x0b,0xe3,0x19]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  19e30be5      <unknown>

rcwcasl  x3, x5, [sp]
// CHECK-INST: rcwcasl x3, x5, [sp]
// CHECK-ENCODING: encoding: [0xe5,0x0b,0x63,0x19]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  19630be5      <unknown>

rcwscas   x0, x1, [x4]
// CHECK-INST: rcwscas x0, x1, [x4]
// CHECK-ENCODING: encoding: [0x81,0x08,0x20,0x59]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  59200881      <unknown>

rcwscasa  x0, x1, [x4]
// CHECK-INST: rcwscasa x0, x1, [x4]
// CHECK-ENCODING: encoding: [0x81,0x08,0xa0,0x59]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  59a00881      <unknown>

rcwscasal x0, x1, [x4]
// CHECK-INST: rcwscasal x0, x1, [x4]
// CHECK-ENCODING: encoding: [0x81,0x08,0xe0,0x59]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  59e00881      <unknown>

rcwscasl  x0, x1, [x4]
// CHECK-INST: rcwscasl x0, x1, [x4]
// CHECK-ENCODING: encoding: [0x81,0x08,0x60,0x59]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  59600881      <unknown>

rcwscas   x3, x5, [sp]
// CHECK-INST: rcwscas x3, x5, [sp]
// CHECK-ENCODING: encoding: [0xe5,0x0b,0x23,0x59]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  59230be5      <unknown>

rcwscasa  x3, x5, [sp]
// CHECK-INST: rcwscasa x3, x5, [sp]
// CHECK-ENCODING: encoding: [0xe5,0x0b,0xa3,0x59]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  59a30be5      <unknown>

rcwscasal x3, x5, [sp]
// CHECK-INST: rcwscasal x3, x5, [sp]
// CHECK-ENCODING: encoding: [0xe5,0x0b,0xe3,0x59]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  59e30be5      <unknown>

rcwscasl  x3, x5, [sp]
// CHECK-INST: rcwscasl x3, x5, [sp]
// CHECK-ENCODING: encoding: [0xe5,0x0b,0x63,0x59]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  59630be5      <unknown>

rcwcasp   x0, x1, x6, x7, [x4]
// CHECK-INST: rcwcasp x0, x1, x6, x7, [x4]
// CHECK-ENCODING: encoding: [0x86,0x0c,0x20,0x19]
// CHECK-ERROR: error: instruction requires: d128 the
// ERROR-NO-D128: error: instruction requires: d128
// CHECK-UNKNOWN:  19200c86      <unknown>

rcwcaspa  x0, x1, x6, x7, [x4]
// CHECK-INST: rcwcaspa x0, x1, x6, x7, [x4]
// CHECK-ENCODING: encoding: [0x86,0x0c,0xa0,0x19]
// CHECK-ERROR: error: instruction requires: d128 the
// ERROR-NO-D128: error: instruction requires: d128
// CHECK-UNKNOWN:  19a00c86      <unknown>

rcwcaspal x0, x1, x6, x7, [x4]
// CHECK-INST: rcwcaspal x0, x1, x6, x7, [x4]
// CHECK-ENCODING: encoding: [0x86,0x0c,0xe0,0x19]
// CHECK-ERROR: error: instruction requires: d128 the
// ERROR-NO-D128: error: instruction requires: d128
// CHECK-UNKNOWN:  19e00c86      <unknown>

rcwcaspl  x0, x1, x6, x7, [x4]
// CHECK-INST: rcwcaspl x0, x1, x6, x7, [x4]
// CHECK-ENCODING: encoding: [0x86,0x0c,0x60,0x19]
// CHECK-ERROR: error: instruction requires: d128 the
// ERROR-NO-D128: error: instruction requires: d128
// CHECK-UNKNOWN:  19600c86      <unknown>

rcwcasp   x4, x5, x6, x7, [sp]
// CHECK-INST: rcwcasp x4, x5, x6, x7, [sp]
// CHECK-ENCODING: encoding: [0xe6,0x0f,0x24,0x19]
// CHECK-ERROR: error: instruction requires: d128 the
// ERROR-NO-D128: error: instruction requires: d128
// CHECK-UNKNOWN:  19240fe6      <unknown>

rcwcaspa  x4, x5, x6, x7, [sp]
// CHECK-INST: rcwcaspa x4, x5, x6, x7, [sp]
// CHECK-ENCODING: encoding: [0xe6,0x0f,0xa4,0x19]
// CHECK-ERROR: error: instruction requires: d128 the
// ERROR-NO-D128: error: instruction requires: d128
// CHECK-UNKNOWN:  19a40fe6      <unknown>

rcwcaspal x4, x5, x6, x7, [sp]
// CHECK-INST: rcwcaspal x4, x5, x6, x7, [sp]
// CHECK-ENCODING: encoding: [0xe6,0x0f,0xe4,0x19]
// CHECK-ERROR: error: instruction requires: d128 the
// ERROR-NO-D128: error: instruction requires: d128
// CHECK-UNKNOWN:  19e40fe6      <unknown>

rcwcaspl  x4, x5, x6, x7, [sp]
// CHECK-INST: rcwcaspl x4, x5, x6, x7, [sp]
// CHECK-ENCODING: encoding: [0xe6,0x0f,0x64,0x19]
// CHECK-ERROR: error: instruction requires: d128 the
// ERROR-NO-D128: error: instruction requires: d128
// CHECK-UNKNOWN:  19640fe6      <unknown>

rcwscasp   x0, x1, x6, x7, [x4]
// CHECK-INST: rcwscasp x0, x1, x6, x7, [x4]
// CHECK-ENCODING: encoding: [0x86,0x0c,0x20,0x59]
// CHECK-ERROR: error: instruction requires: d128 the
// CHECK-UNKNOWN:  59200c86      <unknown>

rcwscaspa  x0, x1, x6, x7, [x4]
// CHECK-INST: rcwscaspa x0, x1, x6, x7, [x4]
// CHECK-ENCODING: encoding: [0x86,0x0c,0xa0,0x59]
// CHECK-ERROR: error: instruction requires: d128 the
// CHECK-UNKNOWN:  59a00c86      <unknown>

rcwscaspal x0, x1, x6, x7, [x4]
// CHECK-INST: rcwscaspal x0, x1, x6, x7, [x4]
// CHECK-ENCODING: encoding: [0x86,0x0c,0xe0,0x59]
// CHECK-ERROR: error: instruction requires: d128 the
// CHECK-UNKNOWN:  59e00c86      <unknown>

rcwscaspl  x0, x1, x6, x7, [x4]
// CHECK-INST: rcwscaspl x0, x1, x6, x7, [x4]
// CHECK-ENCODING: encoding: [0x86,0x0c,0x60,0x59]
// CHECK-ERROR: error: instruction requires: d128 the
// CHECK-UNKNOWN:  59600c86      <unknown>

rcwscasp   x4, x5, x6, x7, [sp]
// CHECK-INST: rcwscasp x4, x5, x6, x7, [sp]
// CHECK-ENCODING: encoding: [0xe6,0x0f,0x24,0x59]
// CHECK-ERROR: error: instruction requires: d128 the
// CHECK-UNKNOWN:  59240fe6      <unknown>

rcwscaspa  x4, x5, x6, x7, [sp]
// CHECK-INST: rcwscaspa x4, x5, x6, x7, [sp]
// CHECK-ENCODING: encoding: [0xe6,0x0f,0xa4,0x59]
// CHECK-ERROR: error: instruction requires: d128 the
// CHECK-UNKNOWN:  59a40fe6      <unknown>

rcwscaspal x4, x5, x6, x7, [sp]
// CHECK-INST: rcwscaspal x4, x5, x6, x7, [sp]
// CHECK-ENCODING: encoding: [0xe6,0x0f,0xe4,0x59]
// CHECK-ERROR: error: instruction requires: d128 the
// CHECK-UNKNOWN:  59e40fe6      <unknown>

rcwscaspl  x4, x5, x6, x7, [sp]
// CHECK-INST: rcwscaspl x4, x5, x6, x7, [sp]
// CHECK-ENCODING: encoding: [0xe6,0x0f,0x64,0x59]
// CHECK-ERROR: error: instruction requires: d128 the
// CHECK-UNKNOWN:  59640fe6      <unknown>

rcwclr   x0, x1, [x4]
// CHECK-INST: rcwclr x0, x1, [x4]
// CHECK-ENCODING: encoding: [0x81,0x90,0x20,0x38]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  38209081      <unknown>

rcwclra  x0, x1, [x4]
// CHECK-INST: rcwclra x0, x1, [x4]
// CHECK-ENCODING: encoding: [0x81,0x90,0xa0,0x38]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  38a09081      <unknown>

rcwclral x0, x1, [x4]
// CHECK-INST: rcwclral x0, x1, [x4]
// CHECK-ENCODING: encoding: [0x81,0x90,0xe0,0x38]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  38e09081      <unknown>

rcwclrl  x0, x1, [x4]
// CHECK-INST: rcwclrl x0, x1, [x4]
// CHECK-ENCODING: encoding: [0x81,0x90,0x60,0x38]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  38609081      <unknown>

rcwclr   x3, x5, [sp]
// CHECK-INST: rcwclr x3, x5, [sp]
// CHECK-ENCODING: encoding: [0xe5,0x93,0x23,0x38]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  382393e5      <unknown>

rcwclra  x3, x5, [sp]
// CHECK-INST: rcwclra x3, x5, [sp]
// CHECK-ENCODING: encoding: [0xe5,0x93,0xa3,0x38]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  38a393e5      <unknown>

rcwclral x3, x5, [sp]
// CHECK-INST: rcwclral x3, x5, [sp]
// CHECK-ENCODING: encoding: [0xe5,0x93,0xe3,0x38]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  38e393e5      <unknown>

rcwclrl  x3, x5, [sp]
// CHECK-INST: rcwclrl x3, x5, [sp]
// CHECK-ENCODING: encoding: [0xe5,0x93,0x63,0x38]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  386393e5      <unknown>

rcwsclr   x0, x1, [x4]
// CHECK-INST: rcwsclr x0, x1, [x4]
// CHECK-ENCODING: encoding: [0x81,0x90,0x20,0x78]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  78209081      <unknown>

rcwsclra  x0, x1, [x4]
// CHECK-INST: rcwsclra x0, x1, [x4]
// CHECK-ENCODING: encoding: [0x81,0x90,0xa0,0x78]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  78a09081      <unknown>

rcwsclral x0, x1, [x4]
// CHECK-INST: rcwsclral x0, x1, [x4]
// CHECK-ENCODING: encoding: [0x81,0x90,0xe0,0x78]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  78e09081      <unknown>

rcwsclrl  x0, x1, [x4]
// CHECK-INST: rcwsclrl x0, x1, [x4]
// CHECK-ENCODING: encoding: [0x81,0x90,0x60,0x78]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  78609081      <unknown>

rcwsclr   x3, x5, [sp]
// CHECK-INST: rcwsclr x3, x5, [sp]
// CHECK-ENCODING: encoding: [0xe5,0x93,0x23,0x78]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  782393e5      <unknown>

rcwsclra  x3, x5, [sp]
// CHECK-INST: rcwsclra x3, x5, [sp]
// CHECK-ENCODING: encoding: [0xe5,0x93,0xa3,0x78]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  78a393e5      <unknown>

rcwsclral x3, x5, [sp]
// CHECK-INST: rcwsclral x3, x5, [sp]
// CHECK-ENCODING: encoding: [0xe5,0x93,0xe3,0x78]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  78e393e5      <unknown>

rcwsclrl  x3, x5, [sp]
// CHECK-INST: rcwsclrl x3, x5, [sp]
// CHECK-ENCODING: encoding: [0xe5,0x93,0x63,0x78]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  786393e5      <unknown>

rcwclrp   x1, x0, [x4]
// CHECK-INST: rcwclrp x1, x0, [x4]
// CHECK-ENCODING: encoding: [0x81,0x90,0x20,0x19]
// CHECK-ERROR: error: instruction requires: d128 the
// ERROR-NO-D128: error: instruction requires: d128
// CHECK-UNKNOWN:  19209081      <unknown>

rcwclrpa  x1, x0, [x4]
// CHECK-INST: rcwclrpa x1, x0, [x4]
// CHECK-ENCODING: encoding: [0x81,0x90,0xa0,0x19]
// CHECK-ERROR: error: instruction requires: d128 the
// ERROR-NO-D128: error: instruction requires: d128
// CHECK-UNKNOWN:  19a09081      <unknown>

rcwclrpal x1, x0, [x4]
// CHECK-INST: rcwclrpal x1, x0, [x4]
// CHECK-ENCODING: encoding: [0x81,0x90,0xe0,0x19]
// CHECK-ERROR: error: instruction requires: d128 the
// ERROR-NO-D128: error: instruction requires: d128
// CHECK-UNKNOWN:  19e09081      <unknown>

rcwclrpl  x1, x0, [x4]
// CHECK-INST: rcwclrpl x1, x0, [x4]
// CHECK-ENCODING: encoding: [0x81,0x90,0x60,0x19]
// CHECK-ERROR: error: instruction requires: d128 the
// ERROR-NO-D128: error: instruction requires: d128
// CHECK-UNKNOWN:  19609081      <unknown>

rcwclrp   x5, x3, [sp]
// CHECK-INST: rcwclrp x5, x3, [sp]
// CHECK-ENCODING: encoding: [0xe5,0x93,0x23,0x19]
// CHECK-ERROR: error: instruction requires: d128 the
// ERROR-NO-D128: error: instruction requires: d128
// CHECK-UNKNOWN:  192393e5      <unknown>

rcwclrpa  x5, x3, [sp]
// CHECK-INST: rcwclrpa x5, x3, [sp]
// CHECK-ENCODING: encoding: [0xe5,0x93,0xa3,0x19]
// CHECK-ERROR: error: instruction requires: d128 the
// ERROR-NO-D128: error: instruction requires: d128
// CHECK-UNKNOWN:  19a393e5      <unknown>

rcwclrpal x5, x3, [sp]
// CHECK-INST: rcwclrpal x5, x3, [sp]
// CHECK-ENCODING: encoding: [0xe5,0x93,0xe3,0x19]
// CHECK-ERROR: error: instruction requires: d128 the
// ERROR-NO-D128: error: instruction requires: d128
// CHECK-UNKNOWN:  19e393e5      <unknown>

rcwclrpl  x5, x3, [sp]
// CHECK-INST: rcwclrpl x5, x3, [sp]
// CHECK-ENCODING: encoding: [0xe5,0x93,0x63,0x19]
// CHECK-ERROR: error: instruction requires: d128 the
// ERROR-NO-D128: error: instruction requires: d128
// CHECK-UNKNOWN:  196393e5      <unknown>

rcwsclrp   x1, x0, [x4]
// CHECK-INST: rcwsclrp x1, x0, [x4]
// CHECK-ENCODING: encoding: [0x81,0x90,0x20,0x59]
// CHECK-ERROR: error: instruction requires: d128 the
// CHECK-UNKNOWN:  59209081      <unknown>

rcwsclrpa  x1, x0, [x4]
// CHECK-INST: rcwsclrpa x1, x0, [x4]
// CHECK-ENCODING: encoding: [0x81,0x90,0xa0,0x59]
// CHECK-ERROR: error: instruction requires: d128 the
// CHECK-UNKNOWN:  59a09081      <unknown>

rcwsclrpal x1, x0, [x4]
// CHECK-INST: rcwsclrpal x1, x0, [x4]
// CHECK-ENCODING: encoding: [0x81,0x90,0xe0,0x59]
// CHECK-ERROR: error: instruction requires: d128 the
// CHECK-UNKNOWN:  59e09081      <unknown>

rcwsclrpl  x1, x0, [x4]
// CHECK-INST: rcwsclrpl x1, x0, [x4]
// CHECK-ENCODING: encoding: [0x81,0x90,0x60,0x59]
// CHECK-ERROR: error: instruction requires: d128 the
// CHECK-UNKNOWN:  59609081      <unknown>

rcwsclrp   x5, x3, [sp]
// CHECK-INST: rcwsclrp x5, x3, [sp]
// CHECK-ENCODING: encoding: [0xe5,0x93,0x23,0x59]
// CHECK-ERROR: error: instruction requires: d128 the
// CHECK-UNKNOWN:  592393e5      <unknown>

rcwsclrpa  x5, x3, [sp]
// CHECK-INST: rcwsclrpa x5, x3, [sp]
// CHECK-ENCODING: encoding: [0xe5,0x93,0xa3,0x59]
// CHECK-ERROR: error: instruction requires: d128 the
// CHECK-UNKNOWN:  59a393e5      <unknown>

rcwsclrpal x5, x3, [sp]
// CHECK-INST: rcwsclrpal x5, x3, [sp]
// CHECK-ENCODING: encoding: [0xe5,0x93,0xe3,0x59]
// CHECK-ERROR: error: instruction requires: d128 the
// CHECK-UNKNOWN:  59e393e5      <unknown>

rcwsclrpl  x5, x3, [sp]
// CHECK-INST: rcwsclrpl x5, x3, [sp]
// CHECK-ENCODING: encoding: [0xe5,0x93,0x63,0x59]
// CHECK-ERROR: error: instruction requires: d128 the
// CHECK-UNKNOWN:  596393e5      <unknown>

rcwset   x0, x1, [x4]
// CHECK-INST: rcwset x0, x1, [x4]
// CHECK-ENCODING: encoding: [0x81,0xb0,0x20,0x38]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  3820b081      <unknown>

rcwseta  x0, x1, [x4]
// CHECK-INST: rcwseta x0, x1, [x4]
// CHECK-ENCODING: encoding: [0x81,0xb0,0xa0,0x38]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  38a0b081      <unknown>

rcwsetal x0, x1, [x4]
// CHECK-INST: rcwsetal x0, x1, [x4]
// CHECK-ENCODING: encoding: [0x81,0xb0,0xe0,0x38]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  38e0b081      <unknown>

rcwsetl  x0, x1, [x4]
// CHECK-INST: rcwsetl x0, x1, [x4]
// CHECK-ENCODING: encoding: [0x81,0xb0,0x60,0x38]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  3860b081      <unknown>

rcwset   x3, x5, [sp]
// CHECK-INST: rcwset x3, x5, [sp]
// CHECK-ENCODING: encoding: [0xe5,0xb3,0x23,0x38]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  3823b3e5      <unknown>

rcwseta  x3, x5, [sp]
// CHECK-INST: rcwseta x3, x5, [sp]
// CHECK-ENCODING: encoding: [0xe5,0xb3,0xa3,0x38]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  38a3b3e5      <unknown>

rcwsetal x3, x5, [sp]
// CHECK-INST: rcwsetal x3, x5, [sp]
// CHECK-ENCODING: encoding: [0xe5,0xb3,0xe3,0x38]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  38e3b3e5      <unknown>

rcwsetl  x3, x5, [sp]
// CHECK-INST: rcwsetl x3, x5, [sp]
// CHECK-ENCODING: encoding: [0xe5,0xb3,0x63,0x38]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  3863b3e5      <unknown>

rcwsset   x0, x1, [x4]
// CHECK-INST: rcwsset x0, x1, [x4]
// CHECK-ENCODING: encoding: [0x81,0xb0,0x20,0x78]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  7820b081      <unknown>

rcwsseta  x0, x1, [x4]
// CHECK-INST: rcwsseta x0, x1, [x4]
// CHECK-ENCODING: encoding: [0x81,0xb0,0xa0,0x78]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  78a0b081      <unknown>

rcwssetal x0, x1, [x4]
// CHECK-INST: rcwssetal x0, x1, [x4]
// CHECK-ENCODING: encoding: [0x81,0xb0,0xe0,0x78]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  78e0b081      <unknown>

rcwssetl  x0, x1, [x4]
// CHECK-INST: rcwssetl x0, x1, [x4]
// CHECK-ENCODING: encoding: [0x81,0xb0,0x60,0x78]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  7860b081      <unknown>

rcwsset   x3, x5, [sp]
// CHECK-INST: rcwsset x3, x5, [sp]
// CHECK-ENCODING: encoding: [0xe5,0xb3,0x23,0x78]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  7823b3e5      <unknown>

rcwsseta  x3, x5, [sp]
// CHECK-INST: rcwsseta x3, x5, [sp]
// CHECK-ENCODING: encoding: [0xe5,0xb3,0xa3,0x78]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  78a3b3e5      <unknown>

rcwssetal x3, x5, [sp]
// CHECK-INST: rcwssetal x3, x5, [sp]
// CHECK-ENCODING: encoding: [0xe5,0xb3,0xe3,0x78]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  78e3b3e5      <unknown>

rcwssetl  x3, x5, [sp]
// CHECK-INST: rcwssetl x3, x5, [sp]
// CHECK-ENCODING: encoding: [0xe5,0xb3,0x63,0x78]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  7863b3e5      <unknown>

rcwsetp   x1, x0, [x4]
// CHECK-INST: rcwsetp x1, x0, [x4]
// CHECK-ENCODING: encoding: [0x81,0xb0,0x20,0x19]
// CHECK-ERROR: error: instruction requires: d128 the
// ERROR-NO-D128: error: instruction requires: d128
// CHECK-UNKNOWN:  1920b081      <unknown>

rcwsetpa  x1, x0, [x4]
// CHECK-INST: rcwsetpa x1, x0, [x4]
// CHECK-ENCODING: encoding: [0x81,0xb0,0xa0,0x19]
// CHECK-ERROR: error: instruction requires: d128 the
// ERROR-NO-D128: error: instruction requires: d128
// CHECK-UNKNOWN:  19a0b081      <unknown>

rcwsetpal x1, x0, [x4]
// CHECK-INST: rcwsetpal x1, x0, [x4]
// CHECK-ENCODING: encoding: [0x81,0xb0,0xe0,0x19]
// CHECK-ERROR: error: instruction requires: d128 the
// ERROR-NO-D128: error: instruction requires: d128
// CHECK-UNKNOWN:  19e0b081      <unknown>

rcwsetpl  x1, x0, [x4]
// CHECK-INST: rcwsetpl x1, x0, [x4]
// CHECK-ENCODING: encoding: [0x81,0xb0,0x60,0x19]
// CHECK-ERROR: error: instruction requires: d128 the
// ERROR-NO-D128: error: instruction requires: d128
// CHECK-UNKNOWN:  1960b081      <unknown>

rcwsetp   x5, x3, [sp]
// CHECK-INST: rcwsetp x5, x3, [sp]
// CHECK-ENCODING: encoding: [0xe5,0xb3,0x23,0x19]
// CHECK-ERROR: error: instruction requires: d128 the
// ERROR-NO-D128: error: instruction requires: d128
// CHECK-UNKNOWN:  1923b3e5      <unknown>

rcwsetpa  x5, x3, [sp]
// CHECK-INST: rcwsetpa x5, x3, [sp]
// CHECK-ENCODING: encoding: [0xe5,0xb3,0xa3,0x19]
// CHECK-ERROR: error: instruction requires: d128 the
// ERROR-NO-D128: error: instruction requires: d128
// CHECK-UNKNOWN:  19a3b3e5      <unknown>

rcwsetpal x5, x3, [sp]
// CHECK-INST: rcwsetpal x5, x3, [sp]
// CHECK-ENCODING: encoding: [0xe5,0xb3,0xe3,0x19]
// CHECK-ERROR: error: instruction requires: d128 the
// ERROR-NO-D128: error: instruction requires: d128
// CHECK-UNKNOWN:  19e3b3e5      <unknown>

rcwsetpl  x5, x3, [sp]
// CHECK-INST: rcwsetpl x5, x3, [sp]
// CHECK-ENCODING: encoding: [0xe5,0xb3,0x63,0x19]
// CHECK-ERROR: error: instruction requires: d128 the
// ERROR-NO-D128: error: instruction requires: d128
// CHECK-UNKNOWN:  1963b3e5      <unknown>

rcwssetp   x1, x0, [x4]
// CHECK-INST: rcwssetp x1, x0, [x4]
// CHECK-ENCODING: encoding: [0x81,0xb0,0x20,0x59]
// CHECK-ERROR: error: instruction requires: d128 the
// ERROR-NO-D128: error: instruction requires: d128
// CHECK-UNKNOWN:  5920b081      <unknown>

rcwssetpa  x1, x0, [x4]
// CHECK-INST: rcwssetpa x1, x0, [x4]
// CHECK-ENCODING: encoding: [0x81,0xb0,0xa0,0x59]
// CHECK-ERROR: error: instruction requires: d128 the
// ERROR-NO-D128: error: instruction requires: d128
// CHECK-UNKNOWN:  59a0b081      <unknown>

rcwssetpal x1, x0, [x4]
// CHECK-INST: rcwssetpal x1, x0, [x4]
// CHECK-ENCODING: encoding: [0x81,0xb0,0xe0,0x59]
// CHECK-ERROR: error: instruction requires: d128 the
// ERROR-NO-D128: error: instruction requires: d128
// CHECK-UNKNOWN:  59e0b081      <unknown>

rcwssetpl  x1, x0, [x4]
// CHECK-INST: rcwssetpl x1, x0, [x4]
// CHECK-ENCODING: encoding: [0x81,0xb0,0x60,0x59]
// CHECK-ERROR: error: instruction requires: d128 the
// ERROR-NO-D128: error: instruction requires: d128
// CHECK-UNKNOWN:  5960b081      <unknown>

rcwssetp   x5, x3, [sp]
// CHECK-INST: rcwssetp x5, x3, [sp]
// CHECK-ENCODING: encoding: [0xe5,0xb3,0x23,0x59]
// CHECK-ERROR: error: instruction requires: d128 the
// ERROR-NO-D128: error: instruction requires: d128
// CHECK-UNKNOWN:  5923b3e5      <unknown>

rcwssetpa  x5, x3, [sp]
// CHECK-INST: rcwssetpa x5, x3, [sp]
// CHECK-ENCODING: encoding: [0xe5,0xb3,0xa3,0x59]
// CHECK-ERROR: error: instruction requires: d128 the
// ERROR-NO-D128: error: instruction requires: d128
// CHECK-UNKNOWN:  59a3b3e5      <unknown>

rcwssetpal x5, x3, [sp]
// CHECK-INST: rcwssetpal x5, x3, [sp]
// CHECK-ENCODING: encoding: [0xe5,0xb3,0xe3,0x59]
// CHECK-ERROR: error: instruction requires: d128 the
// ERROR-NO-D128: error: instruction requires: d128
// CHECK-UNKNOWN:  59e3b3e5      <unknown>

rcwssetpl  x5, x3, [sp]
// CHECK-INST: rcwssetpl x5, x3, [sp]
// CHECK-ENCODING: encoding: [0xe5,0xb3,0x63,0x59]
// CHECK-ERROR: error: instruction requires: d128 the
// ERROR-NO-D128: error: instruction requires: d128
// CHECK-UNKNOWN:  5963b3e5      <unknown>

rcwswp   x0, x1, [x4]
// CHECK-INST: rcwswp x0, x1, [x4]
// CHECK-ENCODING: encoding: [0x81,0xa0,0x20,0x38]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  3820a081      <unknown>

rcwswpa  x0, x1, [x4]
// CHECK-INST: rcwswpa x0, x1, [x4]
// CHECK-ENCODING: encoding: [0x81,0xa0,0xa0,0x38]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  38a0a081      <unknown>

rcwswpal x0, x1, [x4]
// CHECK-INST: rcwswpal x0, x1, [x4]
// CHECK-ENCODING: encoding: [0x81,0xa0,0xe0,0x38]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  38e0a081      <unknown>

rcwswpl  x0, x1, [x4]
// CHECK-INST: rcwswpl x0, x1, [x4]
// CHECK-ENCODING: encoding: [0x81,0xa0,0x60,0x38]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  3860a081      <unknown>

rcwswp   x3, x5, [sp]
// CHECK-INST: rcwswp x3, x5, [sp]
// CHECK-ENCODING: encoding: [0xe5,0xa3,0x23,0x38]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  3823a3e5      <unknown>

rcwswpa  x3, x5, [sp]
// CHECK-INST: rcwswpa x3, x5, [sp]
// CHECK-ENCODING: encoding: [0xe5,0xa3,0xa3,0x38]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  38a3a3e5      <unknown>

rcwswpal x3, x5, [sp]
// CHECK-INST: rcwswpal x3, x5, [sp]
// CHECK-ENCODING: encoding: [0xe5,0xa3,0xe3,0x38]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  38e3a3e5      <unknown>

rcwswpl  x3, x5, [sp]
// CHECK-INST: rcwswpl x3, x5, [sp]
// CHECK-ENCODING: encoding: [0xe5,0xa3,0x63,0x38]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  3863a3e5      <unknown>

rcwsswp   x0, x1, [x4]
// CHECK-INST: rcwsswp x0, x1, [x4]
// CHECK-ENCODING: encoding: [0x81,0xa0,0x20,0x78]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  7820a081      <unknown>

rcwsswpa  x0, x1, [x4]
// CHECK-INST: rcwsswpa x0, x1, [x4]
// CHECK-ENCODING: encoding: [0x81,0xa0,0xa0,0x78]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  78a0a081      <unknown>

rcwsswpal x0, x1, [x4]
// CHECK-INST: rcwsswpal x0, x1, [x4]
// CHECK-ENCODING: encoding: [0x81,0xa0,0xe0,0x78]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  78e0a081      <unknown>

rcwsswpl  x0, x1, [x4]
// CHECK-INST: rcwsswpl x0, x1, [x4]
// CHECK-ENCODING: encoding: [0x81,0xa0,0x60,0x78]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  7860a081      <unknown>

rcwsswp   x3, x5, [sp]
// CHECK-INST: rcwsswp x3, x5, [sp]
// CHECK-ENCODING: encoding: [0xe5,0xa3,0x23,0x78]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  7823a3e5      <unknown>

rcwsswpa  x3, x5, [sp]
// CHECK-INST: rcwsswpa x3, x5, [sp]
// CHECK-ENCODING: encoding: [0xe5,0xa3,0xa3,0x78]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  78a3a3e5      <unknown>

rcwsswpal x3, x5, [sp]
// CHECK-INST: rcwsswpal x3, x5, [sp]
// CHECK-ENCODING: encoding: [0xe5,0xa3,0xe3,0x78]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  78e3a3e5      <unknown>

rcwsswpl  x3, x5, [sp]
// CHECK-INST: rcwsswpl x3, x5, [sp]
// CHECK-ENCODING: encoding: [0xe5,0xa3,0x63,0x78]
// CHECK-ERROR: error: instruction requires: the
// CHECK-UNKNOWN:  7863a3e5      <unknown>

rcwswpp   x1, x0, [x4]
// CHECK-INST: rcwswpp x1, x0, [x4]
// CHECK-ENCODING: encoding: [0x81,0xa0,0x20,0x19]
// CHECK-ERROR: error: instruction requires: d128 the
// CHECK-UNKNOWN:  1920a081      <unknown>

rcwswppa  x1, x0, [x4]
// CHECK-INST: rcwswppa x1, x0, [x4]
// CHECK-ENCODING: encoding: [0x81,0xa0,0xa0,0x19]
// CHECK-ERROR: error: instruction requires: d128 the
// CHECK-UNKNOWN:  19a0a081      <unknown>

rcwswppal x1, x0, [x4]
// CHECK-INST: rcwswppal x1, x0, [x4]
// CHECK-ENCODING: encoding: [0x81,0xa0,0xe0,0x19]
// CHECK-ERROR: error: instruction requires: d128 the
// CHECK-UNKNOWN:  19e0a081      <unknown>

rcwswppl  x1, x0, [x4]
// CHECK-INST: rcwswppl x1, x0, [x4]
// CHECK-ENCODING: encoding: [0x81,0xa0,0x60,0x19]
// CHECK-ERROR: error: instruction requires: d128 the
// CHECK-UNKNOWN:  1960a081      <unknown>

rcwswpp   x5, x3, [sp]
// CHECK-INST: rcwswpp x5, x3, [sp]
// CHECK-ENCODING: encoding: [0xe5,0xa3,0x23,0x19]
// CHECK-ERROR: error: instruction requires: d128 the
// CHECK-UNKNOWN:  1923a3e5      <unknown>

rcwswppa  x5, x3, [sp]
// CHECK-INST: rcwswppa x5, x3, [sp]
// CHECK-ENCODING: encoding: [0xe5,0xa3,0xa3,0x19]
// CHECK-ERROR: error: instruction requires: d128 the
// CHECK-UNKNOWN:  19a3a3e5      <unknown>

rcwswppal x5, x3, [sp]
// CHECK-INST: rcwswppal x5, x3, [sp]
// CHECK-ENCODING: encoding: [0xe5,0xa3,0xe3,0x19]
// CHECK-ERROR: error: instruction requires: d128 the
// CHECK-UNKNOWN:  19e3a3e5      <unknown>

rcwswppl  x5, x3, [sp]
// CHECK-INST: rcwswppl x5, x3, [sp]
// CHECK-ENCODING: encoding: [0xe5,0xa3,0x63,0x19]
// CHECK-ERROR: error: instruction requires: d128 the
// CHECK-UNKNOWN:  1963a3e5      <unknown>

rcwsswpp   x1, x0, [x4]
// CHECK-INST: rcwsswpp x1, x0, [x4]
// CHECK-ENCODING: encoding: [0x81,0xa0,0x20,0x59]
// CHECK-ERROR: error: instruction requires: d128 the
// CHECK-UNKNOWN:  5920a081      <unknown>

rcwsswppa  x1, x0, [x4]
// CHECK-INST: rcwsswppa x1, x0, [x4]
// CHECK-ENCODING: encoding: [0x81,0xa0,0xa0,0x59]
// CHECK-ERROR: error: instruction requires: d128 the
// CHECK-UNKNOWN:  59a0a081      <unknown>

rcwsswppal x1, x0, [x4]
// CHECK-INST: rcwsswppal x1, x0, [x4]
// CHECK-ENCODING: encoding: [0x81,0xa0,0xe0,0x59]
// CHECK-ERROR: error: instruction requires: d128 the
// CHECK-UNKNOWN:  59e0a081      <unknown>

rcwsswppl  x1, x0, [x4]
// CHECK-INST: rcwsswppl x1, x0, [x4]
// CHECK-ENCODING: encoding: [0x81,0xa0,0x60,0x59]
// CHECK-ERROR: error: instruction requires: d128 the
// CHECK-UNKNOWN:  5960a081      <unknown>

rcwsswpp   x5, x3, [sp]
// CHECK-INST: rcwsswpp x5, x3, [sp]
// CHECK-ENCODING: encoding: [0xe5,0xa3,0x23,0x59]
// CHECK-ERROR: error: instruction requires: d128 the
// CHECK-UNKNOWN:  5923a3e5      <unknown>

rcwsswppa  x5, x3, [sp]
// CHECK-INST: rcwsswppa x5, x3, [sp]
// CHECK-ENCODING: encoding: [0xe5,0xa3,0xa3,0x59]
// CHECK-ERROR: error: instruction requires: d128 the
// CHECK-UNKNOWN:  59a3a3e5      <unknown>

rcwsswppal x5, x3, [sp]
// CHECK-INST: rcwsswppal x5, x3, [sp]
// CHECK-ENCODING: encoding: [0xe5,0xa3,0xe3,0x59]
// CHECK-ERROR: error: instruction requires: d128 the
// CHECK-UNKNOWN:  59e3a3e5      <unknown>

rcwsswppl  x5, x3, [sp]
// CHECK-INST: rcwsswppl x5, x3, [sp]
// CHECK-ENCODING: encoding: [0xe5,0xa3,0x63,0x59]
// CHECK-ERROR: error: instruction requires: d128 the
// CHECK-UNKNOWN:  5963a3e5      <unknown>
