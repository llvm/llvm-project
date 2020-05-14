; RUN: opt -codegenprepare -S -mtriple=x86_64 < %s | FileCheck %s

@tmp = global i8 0

; CHECK-LABEL: define void @foo() {
define void @foo() {
enter:
  ; CHECK-NOT: !invariant.group
  ; CHECK-NOT: @llvm.launder.invariant.group.p0i8(
  ; CHECK: %val = load i8, i8* @tmp, align 1{{$}}
  %val = load i8, i8* @tmp, !invariant.group !0
  %ptr = call i8* @llvm.launder.invariant.group.p0i8(i8* @tmp)
  
  ; CHECK: store i8 42, i8* @tmp{{$}}
  store i8 42, i8* %ptr, !invariant.group !0
  
  ret void
}
; CHECK-LABEL: }

; CHECK-LABEL: define void @foo2() {
define void @foo2() {
enter:
  ; CHECK-NOT: !invariant.group
  ; CHECK-NOT: @llvm.strip.invariant.group.p0i8(
  ; CHECK: %val = load i8, i8* @tmp, align 1{{$}}
  %val = load i8, i8* @tmp, !invariant.group !0
  %ptr = call i8* @llvm.strip.invariant.group.p0i8(i8* @tmp)

  ; CHECK: store i8 42, i8* @tmp{{$}}
  store i8 42, i8* %ptr, !invariant.group !0

  ret void
}
; CHECK-LABEL: }


declare i8* @llvm.launder.invariant.group.p0i8(i8*)
declare i8* @llvm.strip.invariant.group.p0i8(i8*)
!0 = !{}
