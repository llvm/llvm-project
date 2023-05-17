; RUN: llc -mtriple=x86_64-apple-darwin -mcpu=corei7                             < %s | FileCheck %s
; RUN: llc -mtriple=x86_64-apple-darwin -mcpu=corei7 -fast-isel -fast-isel-abort=1 < %s | FileCheck %s

; Trivial patchpoint codegen
;
define i64 @trivial_patchpoint_codegen(i64 %p1, i64 %p2, i64 %p3, i64 %p4) {
entry:
; CHECK-LABEL: trivial_patchpoint_codegen:
; CHECK:      movabsq $-559038736, %r11
; CHECK-NEXT: callq *%r11
; CHECK-NEXT: xchgw %ax, %ax
; CHECK:      movq %rax, %[[REG:r.+]]
; CHECK:      callq *%r11
; CHECK-NEXT: xchgw %ax, %ax
; CHECK:      movq %[[REG]], %rax
; CHECK:      ret
  %resolveCall2 = inttoptr i64 -559038736 to ptr
  %result = tail call i64 (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.i64(i64 2, i32 15, ptr %resolveCall2, i32 4, i64 %p1, i64 %p2, i64 %p3, i64 %p4)
  %resolveCall3 = inttoptr i64 -559038737 to ptr
  tail call void (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.void(i64 3, i32 15, ptr %resolveCall3, i32 2, i64 %p1, i64 %result)
  ret i64 %result
}

; Trivial symbolic patchpoint codegen.
;

declare i64 @foo(i64 %p1, i64 %p2)
define i64 @trivial_symbolic_patchpoint_codegen(i64 %p1, i64 %p2) {
entry:
; CHECK-LABEL: trivial_symbolic_patchpoint_codegen:
; CHECK:       movabsq $_foo, %r11
; CHECK-NEXT:  callq   *%r11
; CHECK-NEXT:  xchgw   %ax, %ax
; CHECK:       retq
  %result = tail call i64 (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.i64(i64 9, i32 15, ptr @foo, i32 2, i64 %p1, i64 %p2)
  ret i64 %result
}


; Caller frame metadata with stackmaps. This should not be optimized
; as a leaf function.
;
; CHECK-LABEL: caller_meta_leaf
; CHECK: subq $32, %rsp
; CHECK: Ltmp
; CHECK: addq $32, %rsp
; CHECK: ret
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
  tail call void (i64, i32, ...) @llvm.experimental.stackmap(i64 14, i32 5, i64 %arg, i64 %tmp2, i64 %tmp10, i64 %tmp82)
  tail call void (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.void(i64 15, i32 30, ptr null, i32 3, i64 %arg, i64 %tmp10, i64 %tmp82)
  %tmp83 = load i64, ptr %tmp33, align 8
  %tmp84 = add i64 %tmp83, -24
  %tmp85 = inttoptr i64 %tmp84 to ptr
  %tmp86 = load i64, ptr %tmp85, align 8
  tail call void (i64, i32, ...) @llvm.experimental.stackmap(i64 17, i32 5, i64 %arg, i64 %tmp10, i64 %tmp86)
  tail call void (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.void(i64 18, i32 30, ptr null, i32 3, i64 %arg, i64 %tmp10, i64 %tmp86)
  ret i64 10
}

; Test small patchpoints that don't emit calls.
define void @small_patchpoint_codegen(i64 %p1, i64 %p2, i64 %p3, i64 %p4) {
entry:
; CHECK-LABEL: small_patchpoint_codegen:
; CHECK:      Ltmp
; CHECK:      nopl 8(%rax,%rax)
; CHECK-NEXT: popq
; CHECK-NEXT: ret
  %result = tail call i64 (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.i64(i64 5, i32 5, ptr null, i32 2, i64 %p1, i64 %p2)
  ret void
}

; Test large target address.
define i64 @large_target_address_patchpoint_codegen() {
entry:
; CHECK-LABEL: large_target_address_patchpoint_codegen:
; CHECK:      movabsq $6153737369414576827, %r11
; CHECK-NEXT: callq *%r11
  %resolveCall2 = inttoptr i64 6153737369414576827 to ptr
  %result = tail call i64 (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.i64(i64 2, i32 15, ptr %resolveCall2, i32 0)
  ret i64 %result
}

declare i64 @consume_attributes(i64, ptr nest, i64)
define i64 @test_patchpoint_with_attributes() {
entry:
; CHECK-LABEL: test_patchpoint_with_attributes:
; CHECK: movl $42, %edi
; CHECK: movl $17, %esi
; CHECK: xorl %r10d, %r10d
; CHECK: movabsq $_consume_attributes, %r11
; CHECK-NEXT: callq *%r11
; CHECK-NEXT: xchgw %ax, %ax
; CHECK: retq
  %result = tail call i64 (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.i64(i64 21, i32 15, ptr @consume_attributes, i32 3, i64 42, ptr nest null, i64 17)
  ret i64 %result
}

declare void @llvm.experimental.stackmap(i64, i32, ...)
declare void @llvm.experimental.patchpoint.void(i64, i32, ptr, i32, ...)
declare i64 @llvm.experimental.patchpoint.i64(i64, i32, ptr, i32, ...)
