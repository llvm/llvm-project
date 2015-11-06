; RUN: llc -verify-machineinstrs < %s -mtriple=x86_64-unknown-unknown | FileCheck %s
; RUN: llc -O0 -verify-machineinstrs < %s -mtriple=x86_64-unknown-unknown | FileCheck --check-prefix=CHECK-O0 %s

; Parameter with swiftself should be allocated to r10.
define void @check_swiftself(i32* swiftself %addr0) {
; CHECK-LABEL: check_swiftself:
; CHECK-O0-LABEL: check_swiftself:

  %val0 = load volatile i32, i32* %addr0
; CHECK: movl (%r10),
; CHECK-O0: movl (%r10),
  ret void
}

@var8_3 = global i8 0
declare void @take_swiftself(i8* swiftself %addr0)

define void @simple_args() {
; CHECK-LABEL: simple_args:
; CHECK-O0-LABEL: simple_args:

  call void @take_swiftself(i8* @var8_3)
; CHECK: movl {{.*}}, %r10d
; CHECK: callq {{_?}}take_swiftself
; CHECK-O0: movabsq {{.*}}, %r10
; CHECK-O0: callq {{_?}}take_swiftself

  ret void
}
