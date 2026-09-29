; RUN: llc -mtriple=s390x-linux-gnu < %s | FileCheck %s

; Trivial patchpoint codegen
;
define i64 @trivial_patchpoint_codegen(i64 %p1, i64 %p2, i64 %p3, i64 %p4) {
entry:
; CHECK-LABEL: trivial_patchpoint_codegen:
; CHECK:       llilf   %r1, 559038736
; CHECK-NEXT:  basr    %r14, %r1
; CHECK-NEXT:  bcr     0, %r0
; CHECK:       lgr     [[REG0:%r[0-9]+]], %r2
; CHECK:       llilf   %r1, 559038737
; CHECK-NEXT:  basr    %r14, %r1
; CHECK-NEXT:  bcr     0, %r0
; CHECK:       lgr     %r2, [[REG0:%r[0-9]+]]
; CHECK:       br      %r14
  %resolveCall2 = inttoptr i64 559038736 to ptr
  %result = tail call i64 (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.i64(i64 2, i32 10, ptr %resolveCall2, i32 4, i64 %p1, i64 %p2, i64 %p3, i64 %p4)
  %resolveCall3 = inttoptr i64 559038737 to ptr
  tail call void (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.void(i64 3, i32 10, ptr %resolveCall3, i32 2, i64 %p1, i64 %result)
  ret i64 %result
}

; Trivial symbolic patchpoint codegen.
;

declare i64 @foo(i64 %p1, i64 %p2)
define i64 @trivial_symbolic_patchpoint_codegen(i64 %p1, i64 %p2) {
entry:
; CHECK-LABEL: trivial_symbolic_patchpoint_codegen:
; CHECK:       brasl   %r14, foo@PLT
; CHECK-NEXT:  bcr     0, %r0
; CHECK:       br      %r14
  %result = tail call i64 (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.i64(i64 9, i32 8, ptr @foo, i32 2, i64 %p1, i64 %p2)
  ret i64 %result
}


; Caller frame metadata with stackmaps. This should not be optimized
; as a leaf function.
;
; CHECK-LABEL: caller_meta_leaf
; CHECK: aghi  %r15, -184
; CHECK: .Ltmp
; CHECK: lmg   %r14, %r15, 296(%r15)
; CHECK: br    %r14
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
  tail call void (i64, i32, ...) @llvm.experimental.stackmap(i64 14, i32 6, i64 %arg, i64 %tmp2, i64 %tmp10, i64 %tmp82)
  tail call void (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.void(i64 15, i32 30, ptr null, i32 3, i64 %arg, i64 %tmp10, i64 %tmp82)
  %tmp83 = load i64, ptr %tmp33, align 8
  %tmp84 = add i64 %tmp83, -24
  %tmp85 = inttoptr i64 %tmp84 to ptr
  %tmp86 = load i64, ptr %tmp85, align 8
  tail call void (i64, i32, ...) @llvm.experimental.stackmap(i64 17, i32 6, i64 %arg, i64 %tmp10, i64 %tmp86)
  tail call void (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.void(i64 18, i32 30, ptr null, i32 3, i64 %arg, i64 %tmp10, i64 %tmp86)
  ret i64 10
}

; Test small patchpoints that don't emit calls.
define void @small_patchpoint_codegen(i64 %p1, i64 %p2, i64 %p3, i64 %p4) {
entry:
; CHECK-LABEL: small_patchpoint_codegen:
; CHECK:      .Ltmp
; CHECK:      bcr 0, %r0
; CHECK:      br %r14
  %result = tail call i64 (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.i64(i64 5, i32 2, ptr null, i32 2, i64 %p1, i64 %p2)
  ret void
}

; Test large target address.
define i64 @large_target_address_patchpoint_codegen() {
entry:
; CHECK-LABEL: large_target_address_patchpoint_codegen:
; CHECK:        llilf   %r1, 2566957755
; CHECK-NEXT:   iihf    %r1, 1432778632
; CHECK-NEXT:   basr    %r14, %r1
  %resolveCall2 = inttoptr i64 6153737369414576827 to ptr
  %result = tail call i64 (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.i64(i64 2, i32 14, ptr %resolveCall2, i32 0)
  ret i64 %result
}

; Test that the number of bytes is reflected in the instruction size and
; therefore cause relaxation of the initial branch.
define void @patchpoint_size(i32 %Arg) {
; CHECK-LABEL: patchpoint_size:
; CHECK: # %bb.0:
; CHECK-NEXT: stmg    %r14, %r15, 112(%r15)
; CHECK-NEXT: .cfi_offset %r14, -48
; CHECK-NEXT: .cfi_offset %r15, -40
; CHECK-NEXT: aghi    %r15, -160
; CHECK-NEXT: .cfi_def_cfa_offset 320
; CHECK-NEXT: chi     %r2, 0
; CHECK-NEXT: jge     .LBB6_2
  %c = icmp eq i32 %Arg, 0
  br i1 %c, label %block0, label %patch1

block0:
  call i64 @foo(i64 0, i64 0)
  br label %exit

patch1:
  call void (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.void(i64 0, i32 65536, ptr null, i32 0)
  br label %exit

exit:
  ret void
}

define void @stackmap_size(i32 %Arg) {
; CHECK-LABEL: stackmap_size:
; CHECK: # %bb.0:
; CHECK-NEXT: stmg    %r14, %r15, 112(%r15)
; CHECK-NEXT: .cfi_offset %r14, -48
; CHECK-NEXT: .cfi_offset %r15, -40
; CHECK-NEXT: aghi    %r15, -160
; CHECK-NEXT: .cfi_def_cfa_offset 320
; CHECK-NEXT: chi     %r2, 0
; CHECK-NEXT: jge     .LBB7_2
  %c = icmp eq i32 %Arg, 0
  br i1 %c, label %block0, label %stackmap1

block0:
  call i64 @foo(i64 0, i64 0)
  br label %exit

stackmap1:
  call void (i64, i32, ...) @llvm.experimental.stackmap(i64 1, i32 65536)
  br label %exit

exit:
  ret void
}


declare void @llvm.experimental.stackmap(i64, i32, ...)
declare void @llvm.experimental.patchpoint.void(i64, i32, ptr, i32, ...)
declare i64 @llvm.experimental.patchpoint.i64(i64, i32, ptr, i32, ...)
