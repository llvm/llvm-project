// RUN: not llvm-mc -triple arm64 -show-encoding < %s 2>%t | FileCheck %s
// RUN: FileCheck --check-prefix=CHECK-ERRORS < %t %s

adr x0, #0
adr x0, #1
adr x0, 1f
adr x0, foo
// CHECK: adr x0, #0          // encoding: [0x00,0x00,0x00,0x10]
// CHECK: adr x0, #1          // encoding: [0x00,0x00,0x00,0x30]
// CHECK: adr x0, .Ltmp0      // encoding: [A,A,A,0x10'A']
// CHECK-NEXT:                //   fixup A - offset: 0, value: .Ltmp0, kind: fixup_aarch64_pcrel_adr_imm21
// CHECK: adr x0, foo         // encoding: [A,A,A,0x10'A']
// CHECK-NEXT:                //   fixup A - offset: 0, value: foo, kind: fixup_aarch64_pcrel_adr_imm21

adrp x0, #0
adrp x0, #4096
adrp x0, 1f
adrp x0, foo
// CHECK: adrp    x0, #0      // encoding: [0x00,0x00,0x00,0x90]
// CHECK: adrp    x0, #4096   // encoding: [0x00,0x00,0x00,0xb0]
// CHECK: adrp    x0, .Ltmp0  // encoding: [A,A,A,0x90'A']
// CHECK-NEXT:                //   fixup A - offset: 0, value: .Ltmp0, kind: fixup_aarch64_pcrel_adrp_imm21
// CHECK: adrp    x0, foo     // encoding: [A,A,A,0x90'A']
// CHECK-NEXT:                //   fixup A - offset: 0, value: foo, kind: fixup_aarch64_pcrel_adrp_imm21

adr x0, #0xffffffff
adrp x0, #0xffffffff
adrp x0, #1
// CHECK-ERRORS: error: expected label or encodable integer pc offset
// CHECK-ERRORS: error: expected label or encodable integer pc offset
// CHECK-ERRORS: error: expected label or encodable integer pc offset
