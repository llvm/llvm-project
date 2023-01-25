; RUN: opt -codegenprepare -S -mtriple=x86_64 < %s | FileCheck %s

@tmp = global i8 0

; CHECK-LABEL: define void @foo() {
define void @foo() {
enter:
  ; CHECK-NOT: !invariant.group
  ; CHECK-NOT: @llvm.launder.invariant.group.p0(
  ; CHECK: %val = load i8, ptr @tmp, align 1{{$}}
  %val = load i8, ptr @tmp, !invariant.group !0
  %ptr = call ptr @llvm.launder.invariant.group.p0(ptr @tmp)
  
  ; CHECK: store i8 42, ptr @tmp, align 1{{$}}
  store i8 42, ptr %ptr, !invariant.group !0
  
  ret void
}
; CHECK-LABEL: }

; CHECK-LABEL: define void @foo2() {
define void @foo2() {
enter:
  ; CHECK-NOT: !invariant.group
  ; CHECK-NOT: @llvm.strip.invariant.group.p0(
  ; CHECK: %val = load i8, ptr @tmp, align 1{{$}}
  %val = load i8, ptr @tmp, !invariant.group !0
  %ptr = call ptr @llvm.strip.invariant.group.p0(ptr @tmp)

  ; CHECK: store i8 42, ptr @tmp, align 1{{$}}
  store i8 42, ptr %ptr, !invariant.group !0

  ret void
}
; CHECK-LABEL: }


declare ptr @llvm.launder.invariant.group.p0(ptr)
declare ptr @llvm.strip.invariant.group.p0(ptr)
!0 = !{}
