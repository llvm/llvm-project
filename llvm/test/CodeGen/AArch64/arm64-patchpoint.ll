; RUN: llc -mtriple=arm64-apple-darwin -debug-entry-values -enable-misched=0 -mcpu=cyclone                             < %s | FileCheck %s
; RUN: llc -mtriple=arm64-apple-darwin -debug-entry-values -enable-misched=0 -mcpu=cyclone -fast-isel -fast-isel-abort=1 < %s | FileCheck %s

; Trivial patchpoint codegen
;
define i64 @trivial_patchpoint_codegen(i64 %p1, i64 %p2, i64 %p3, i64 %p4) {
entry:
; CHECK-LABEL: trivial_patchpoint_codegen:
; CHECK:       mov  x16, #244834610708480
; CHECK-NEXT:  movk x16, #48879, lsl #16
; CHECK-NEXT:  movk x16, #51966
; CHECK-NEXT:  blr  x16
; CHECK:       mov  x16, #244834610708480
; CHECK-NEXT:  movk x16, #48879, lsl #16
; CHECK-NEXT:  movk x16, #51967
; CHECK-NEXT:  blr  x16
; CHECK:       ret
  %resolveCall2 = inttoptr i64 244837814094590 to ptr
  %result = tail call i64 (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.i64(i64 2, i32 20, ptr %resolveCall2, i32 4, i64 %p1, i64 %p2, i64 %p3, i64 %p4)
  %resolveCall3 = inttoptr i64 244837814094591 to ptr
  tail call void (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.void(i64 3, i32 20, ptr %resolveCall3, i32 2, i64 %p1, i64 %result)
  ret i64 %result
}

; Caller frame metadata with stackmaps. This should not be optimized
; as a leaf function.
;
; CHECK-LABEL: caller_meta_leaf
; CHECK:       sub sp, sp, #48
; CHECK-NEXT:  stp x29, x30, [sp, #32]
; CHECK-NEXT:  add x29, sp, #32
; CHECK:       Ltmp
; CHECK:       add sp, sp, #48
; CHECK:       ret

define void @caller_meta_leaf() {
entry:
  %metadata = alloca i64, i32 3, align 8
  store i64 11, ptr %metadata
  store i64 12, ptr %metadata
  store i64 13, ptr %metadata
  call void (i64, i32, ...) @llvm.experimental.stackmap(i64 4, i32 0, ptr %metadata)
  ret void
}

; Test patchpoints reusing the same TargetConstant.
; <rdar:15390785> Assertion failed: (CI.getNumArgOperands() >= NumArgs + 4)
; There is no way to verify this, since it depends on memory allocation.
; But I think it's useful to include as a working example.
define i64 @testLowerConstant(i64 %arg, i64 %tmp2, i64 %tmp10, ptr %tmp33, i64 %tmp79) {
entry:
  %tmp80 = add i64 %tmp79, -16
  %tmp81 = inttoptr i64 %tmp80 to ptr
  %tmp82 = load i64, ptr %tmp81, align 8
  tail call void (i64, i32, ...) @llvm.experimental.stackmap(i64 14, i32 8, i64 %arg, i64 %tmp2, i64 %tmp10, i64 %tmp82)
  tail call void (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.void(i64 15, i32 32, ptr null, i32 3, i64 %arg, i64 %tmp10, i64 %tmp82)
  %tmp83 = load i64, ptr %tmp33, align 8
  %tmp84 = add i64 %tmp83, -24
  %tmp85 = inttoptr i64 %tmp84 to ptr
  %tmp86 = load i64, ptr %tmp85, align 8
  tail call void (i64, i32, ...) @llvm.experimental.stackmap(i64 17, i32 8, i64 %arg, i64 %tmp10, i64 %tmp86)
  tail call void (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.void(i64 18, i32 32, ptr null, i32 3, i64 %arg, i64 %tmp10, i64 %tmp86)
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
; CHECK-NEXT: ldp
; CHECK-NEXT: ret
  %result = tail call i64 (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.i64(i64 5, i32 20, ptr null, i32 2, i64 %p1, i64 %p2)
  ret void
}

; Test register allocation for an i32 result value of patchpoint.
define i32 @generic_patchpoint_i32() {
entry:
; CHECK-LABEL: generic_patchpoint_i32:
; CHECK:      Ltmp
; CHECK-NEXT: nop
; The return value is already in w0.
; CHECK-NEXT: ldp
; CHECK-NEXT: ret
  %result = tail call i32 (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.i32(i64 5, i32 4, ptr null, i32 0)
  ret i32 %result
}

; Test register allocation for an i64 result value of patchpoint.
define i64 @generic_patchpoint_i64() {
entry:
; CHECK-LABEL: generic_patchpoint_i64:
; CHECK:      Ltmp
; CHECK-NEXT: nop
; The return value is already in x0.
; CHECK-NEXT: ldp
; CHECK-NEXT: ret
  %result = tail call i64 (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.i64(i64 5, i32 4, ptr null, i32 0)
  ret i64 %result
}

; Test register allocation for a float result value of patchpoint.
define float @generic_patchpoint_f32() {
entry:
; CHECK-LABEL: generic_patchpoint_f32:
; CHECK:      Ltmp
; CHECK-NEXT: nop
; The return value is already in s0.
; CHECK-NEXT: ldp
; CHECK-NEXT: ret
  %result = tail call float (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.f32(i64 5, i32 4, ptr null, i32 0)
  ret float %result
}

; Test register allocation for a double result value of patchpoint.
define double @generic_patchpoint_f64() {
entry:
; CHECK-LABEL: generic_patchpoint_f64:
; CHECK:      Ltmp
; CHECK-NEXT: nop
; The return value is already in d0.
; CHECK-NEXT: ldp
; CHECK-NEXT: ret
  %result = tail call double (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.f64(i64 5, i32 4, ptr null, i32 0)
  ret double %result
}

declare void @llvm.experimental.stackmap(i64, i32, ...)
declare void @llvm.experimental.patchpoint.void(i64, i32, ptr, i32, ...)
declare i32 @llvm.experimental.patchpoint.i32(i64, i32, ptr, i32, ...)
declare i64 @llvm.experimental.patchpoint.i64(i64, i32, ptr, i32, ...)
declare float @llvm.experimental.patchpoint.f32(i64, i32, ptr, i32, ...)
declare double @llvm.experimental.patchpoint.f64(i64, i32, ptr, i32, ...)
