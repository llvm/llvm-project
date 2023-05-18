; RUN: opt -S -passes=inline %s -debug-only=inline-cost 2>&1 | FileCheck %s
; REQUIRES: asserts

@a = global i32 4

; Only the load and ret should be included in the instruction count, not
; the instructions feeding the assume.
; CHECK: Analyzing call of inner...
; CHECK: NumInstructions: 2
define i32 @inner(ptr %y) {
  %a1 = load volatile i32, ptr @a

  ; Because these instructions are used only by the @llvm.assume intrinsic,
  ; they're free and should not be included in the instruction count when
  ; computing the inline cost.
  %a2 = mul i32 %a1, %a1
  %a3 = sub i32 %a1, %a2
  %a4 = udiv i32 %a3, -13
  %a5 = mul i32 %a4, %a4
  %a6 = add i32 %a5, %a5
  %ca = icmp sgt i32 %a6, -7
  %r = call i1 @llvm.type.test(ptr %y, metadata !0)
  %ca2 = icmp eq i1 %ca, %r
  tail call void @llvm.assume(i1 %ca2)

  ret i32 %a1
}

; Only the ret should be included in the instruction count, the load and icmp
; are both ephemeral.
; CHECK: Analyzing call of inner2...
; CHECK: NumInstructions: 1
define void @inner2(ptr %y) {
  %v = load i8, ptr %y
  %c = icmp eq i8 %v, 42
  call void @llvm.assume(i1 %c)
  ret void
}

define i32 @outer(ptr %y) optsize {
   %r = call i32 @inner(ptr %y)
   call void @inner2(ptr %y)
   ret i32 %r
}

declare void @llvm.assume(i1) nounwind
declare i1 @llvm.type.test(ptr, metadata) nounwind readnone

!0 = !{i32 0, !"typeid1"}
