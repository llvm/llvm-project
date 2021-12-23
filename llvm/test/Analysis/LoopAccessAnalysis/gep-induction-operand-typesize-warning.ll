; RUN: opt -loop-load-elim -mtriple=aarch64--linux-gnu -mattr=+sve < %s

; This regression test is verifying that a GEP instruction performed on a
; scalable vector does not produce a 'assumption that TypeSize is not scalable'
; warning in the llvm::getGEPInductionOperand function.

define void @get_gep_induction_operand_typesize_warning(i64 %n, <vscale x 4 x i32>* %a) {
entry:
  br label %loop.body

loop.body:
  %0 = phi i64 [ 0, %entry ], [ %1, %loop.body ]
  %idx = getelementptr <vscale x 4 x i32>, <vscale x 4 x i32>* %a, i64 %0
  store <vscale x 4 x i32> zeroinitializer, <vscale x 4 x i32>* %idx
  %1 = add i64 %0, 1
  %2 = icmp eq i64 %1, %n
  br i1 %2, label %loop.end, label %loop.body

loop.end:
  ret void
}
