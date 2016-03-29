; RUN: llc -verify-machineinstrs < %s -mtriple=x86_64-unknown-unknown | FileCheck %s
; RUN: llc -O0 -verify-machineinstrs < %s -mtriple=x86_64-unknown-unknown | FileCheck --check-prefix=CHECK-O0 %s
; RUN: llc -verify-machineinstrs < %s -march=x86 -mcpu=yonah -mtriple=i386-apple-darwin | FileCheck --check-prefix=CHECK-i386 %s
; RUN: llc -O0 -verify-machineinstrs < %s -march=x86 -mcpu=yonah -mtriple=i386-apple-darwin | FileCheck --check-prefix=CHECK-i386-O0 %s

; Parameter with swiftself should be allocated to r10.
define void @check_swiftself(i32* swiftself %addr0) {
; CHECK-LABEL: check_swiftself:
; CHECK-O0-LABEL: check_swiftself:
; CHECK-i386-LABEL: check_swiftself:
; CHECK-i386-O0-LABEL: check_swiftself:

  %val0 = load volatile i32, i32* %addr0
; CHECK: movl (%r10),
; CHECK-O0: movl (%r10),
; CHECK-i386: movl {{[0-9a-f]+}}(%esp)
; CHECK-i386-O0: movl {{[0-9a-f]+}}(%esp)
  ret void
}

@var8_3 = global i8 0
declare void @take_swiftself(i8* swiftself %addr0)

define void @simple_args() {
; CHECK-LABEL: simple_args:
; CHECK-O0-LABEL: simple_args:
; CHECK-i386-LABEL: simple_args:
; CHECK-i386-O0-LABEL: simple_args:

  call void @take_swiftself(i8* @var8_3)
; CHECK: movl {{.*}}, %r10d
; CHECK: callq {{_?}}take_swiftself
; CHECK-O0: movabsq {{.*}}, %r10
; CHECK-O0: callq {{_?}}take_swiftself
; CHECK-i386: movl {{.*}}, (%esp)
; CHECK-i386: calll {{.*}}take_swiftself
; CHECK-i386-O0: movl {{.*}}, (%esp)
; CHECK-i386-O0: calll {{.*}}take_swiftself

  ret void
}
