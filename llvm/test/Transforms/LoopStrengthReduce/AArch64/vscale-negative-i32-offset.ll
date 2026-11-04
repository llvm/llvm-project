; REQUIRES: asserts
; RUN: opt %s -disable-output -passes=loop-reduce -S -debug-only=loop-reduce 2>&1 | FileCheck %s

target triple = "aarch64-unknown-linux-gnu"

; Regression test to ensure that a negative scalable offset like `-4 * vscale`
; with a type of i32 is correctly sign extended to 64-bits in ExtractImmediateOperand.

; CHECK-NOT: vscale x 4294967292
define void @vscale_neg_offset(ptr %ptr, i32 %0, i32 %n) #0 {
entry:
  %vscale = tail call i32 @llvm.vscale.i32()
  %start = mul nuw i32 %vscale, -4
  %off = shl i32 %vscale, 4
  br label %vector.body

vector.body:
  %iv = phi i32 [ %start, %entry ], [ %iv.next, %vector.body ]
  %gep.part0 = getelementptr [4 x i8], ptr %ptr, i32 %iv
  %data = load <vscale x 4 x i32>, ptr %gep.part0, align 4
  store <vscale x 4 x i32> %data, ptr %gep.part0, align 4
  %iv.next = add nuw i32 %iv, 1
  %exit.cond = icmp eq i32 %iv.next, %n
  br i1 %exit.cond, label %exit, label %vector.body

exit:
  ret void
}

attributes #0 = { "target-features"="+sve" }
