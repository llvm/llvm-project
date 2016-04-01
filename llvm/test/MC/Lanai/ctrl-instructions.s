// RUN: llvm-mc -triple lanai-unknown-unknown -show-encoding -o - %s | FileCheck %s

// CHECK: bt .Ltmp0 ! encoding: [0b1110000A,A,A,A]
// CHECK-NEXT:      ! fixup A - offset: 0, value: .Ltmp0, kind: FIXUP_LANAI_25
  bt 1f
  nop
1:

// CHECK: bt foo    ! encoding: [0b1110000A,A,A,A]
// CHECK-NEXT:      !   fixup A - offset: 0, value: foo, kind: FIXUP_LANAI_25
  bt foo
  nop

