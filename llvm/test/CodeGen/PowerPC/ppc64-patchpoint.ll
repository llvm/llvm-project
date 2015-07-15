; RUN: llc                             < %s | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-BE
; RUN: llc -fast-isel -fast-isel-abort=1 < %s | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-BE
; RUN: llc -mtriple=powerpc64le-unknown-linux-gnu                             < %s | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-LE
; RUN: llc -mtriple=powerpc64le-unknown-linux-gnu -fast-isel -fast-isel-abort=1 < %s | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-LE

target triple = "powerpc64-unknown-linux-gnu"

; Trivial patchpoint codegen
;
define i64 @trivial_patchpoint_codegen(i64 %p1, i64 %p2, i64 %p3, i64 %p4) {
entry:
; CHECK-LABEL: trivial_patchpoint_codegen:

; CHECK: li 12, -8531
; CHECK-NEXT: rldic 12, 12, 32, 16
; CHECK-NEXT: oris 12, 12, 48879
; CHECK-NEXT: ori 12, 12, 51966
; CHECK-LE-NEXT: std 2, 24(1)
; CHECK-BE-NEXT: std 2, 40(1)
; CHECK-BE-NEXT: ld 2, 8(12)
; CHECK-BE-NEXT: ld 12, 0(12)
; CHECK-NEXT: mtctr 12
; CHECK-NEXT: bctrl
; CHECK-LE-NEXT: ld 2, 24(1)
; CHECK-BE-NEXT: ld 2, 40(1)

; CHECK: li 12, -8531
; CHECK-NEXT: rldic 12, 12, 32, 16
; CHECK-NEXT: oris 12, 12, 48879
; CHECK-NEXT: ori 12, 12, 51967
; CHECK-LE-NEXT: std 2, 24(1)
; CHECK-BE-NEXT: std 2, 40(1)
; CHECK-BE-NEXT: ld 2, 8(12)
; CHECK-BE-NEXT: ld 12, 0(12)
; CHECK-NEXT: mtctr 12
; CHECK-NEXT: bctrl
; CHECK-LE-NEXT: ld 2, 24(1)
; CHECK-BE-NEXT: ld 2, 40(1)

; CHECK: blr

  %resolveCall2 = inttoptr i64 244837814094590 to i8*
  %result = tail call i64 (i64, i32, i8*, i32, ...) @llvm.experimental.patchpoint.i64(i64 2, i32 40, i8* %resolveCall2, i32 4, i64 %p1, i64 %p2, i64 %p3, i64 %p4)
  %resolveCall3 = inttoptr i64 244837814094591 to i8*
  tail call void (i64, i32, i8*, i32, ...) @llvm.experimental.patchpoint.void(i64 3, i32 40, i8* %resolveCall3, i32 2, i64 %p1, i64 %result)
  ret i64 %result
}

; Caller frame metadata with stackmaps. This should not be optimized
; as a leaf function.
;
; CHECK-LABEL: caller_meta_leaf
; CHECK-BE: stdu 1, -80(1)
; CHECK-LE: stdu 1, -64(1)
; CHECK: Ltmp
; CHECK-BE: addi 1, 1, 80
; CHECK-LE: addi 1, 1, 64
; CHECK: blr

define void @caller_meta_leaf() {
entry:
  %metadata = alloca i64, i32 3, align 8
  store i64 11, i64* %metadata
  store i64 12, i64* %metadata
  store i64 13, i64* %metadata
  call void (i64, i32, ...) @llvm.experimental.stackmap(i64 4, i32 0, i64* %metadata)
  ret void
}

; Test patchpoints reusing the same TargetConstant.
; <rdar:15390785> Assertion failed: (CI.getNumArgOperands() >= NumArgs + 4)
; There is no way to verify this, since it depends on memory allocation.
; But I think it's useful to include as a working example.
define i64 @testLowerConstant(i64 %arg, i64 %tmp2, i64 %tmp10, i64* %tmp33, i64 %tmp79) {
entry:
  %tmp80 = add i64 %tmp79, -16
  %tmp81 = inttoptr i64 %tmp80 to i64*
  %tmp82 = load i64, i64* %tmp81, align 8
  tail call void (i64, i32, ...) @llvm.experimental.stackmap(i64 14, i32 8, i64 %arg, i64 %tmp2, i64 %tmp10, i64 %tmp82)
  tail call void (i64, i32, i8*, i32, ...) @llvm.experimental.patchpoint.void(i64 15, i32 48, i8* null, i32 3, i64 %arg, i64 %tmp10, i64 %tmp82)
  %tmp83 = load i64, i64* %tmp33, align 8
  %tmp84 = add i64 %tmp83, -24
  %tmp85 = inttoptr i64 %tmp84 to i64*
  %tmp86 = load i64, i64* %tmp85, align 8
  tail call void (i64, i32, ...) @llvm.experimental.stackmap(i64 17, i32 8, i64 %arg, i64 %tmp10, i64 %tmp86)
  tail call void (i64, i32, i8*, i32, ...) @llvm.experimental.patchpoint.void(i64 18, i32 48, i8* null, i32 3, i64 %arg, i64 %tmp10, i64 %tmp86)
  ret i64 10
}

; Test small patchpoints that don't emit calls.
define void @small_patchpoint_codegen(i64 %p1, i64 %p2, i64 %p3, i64 %p4) {
entry:
; CHECK-LABEL: small_patchpoint_codegen:
; CHECK:      Ltmp
; CHECK:      nop
; CHECK-NEXT: nop
; CHECK-NEXT: nop
; CHECK-NEXT: nop
; CHECK-NEXT: nop
; CHECK-NOT:  nop
; CHECK: blr
  %result = tail call i64 (i64, i32, i8*, i32, ...) @llvm.experimental.patchpoint.i64(i64 5, i32 20, i8* null, i32 2, i64 %p1, i64 %p2)
  ret void
}

; Trivial symbolic patchpoint codegen.

declare i64 @foo(i64 %p1, i64 %p2)
define i64 @trivial_symbolic_patchpoint_codegen(i64 %p1, i64 %p2) {
entry:
; CHECK-LABEL: trivial_symbolic_patchpoint_codegen:
; CHECK:       bl foo
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NOT:   nop
; CHECK:       blr
  %result = tail call i64 (i64, i32, i8*, i32, ...) @llvm.experimental.patchpoint.i64(i64 9, i32 12, i8* bitcast (i64 (i64, i64)* @foo to i8*), i32 2, i64 %p1, i64 %p2)
  ret i64 %result
}

declare void @llvm.experimental.stackmap(i64, i32, ...)
declare void @llvm.experimental.patchpoint.void(i64, i32, i8*, i32, ...)
declare i64 @llvm.experimental.patchpoint.i64(i64, i32, i8*, i32, ...)

