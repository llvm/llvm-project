; RUN: llc --mtriple=loongarch32 < %s
; RUN: llc --mtriple=loongarch64 < %s

;; This should not crash the code generator.

define void @_ZN12_GLOBAL__N_111DumpVisitorclIN4llvm16itanium_demangle8FoldExprEEEvPKT_() {
entry:
  %ref.tmp6.i.i = alloca [4 x i8], align 1
  br label %for.cond.i.i

for.cond.i.i:                                     ; preds = %for.body.i.i, %entry
  %__begin0.0.add.i.i = add nuw nsw i64 poison, 1
  br label %for.body.i.i

for.body.i.i:                                     ; preds = %for.cond.i.i
  %__begin0.0.ptr.i.i = getelementptr inbounds i8, ptr %ref.tmp6.i.i, i64 %__begin0.0.add.i.i
  %0 = load i8, ptr %__begin0.0.ptr.i.i, align 1
  %tobool18.not.i.i = icmp eq i8 %0, 0
  br i1 %tobool18.not.i.i, label %for.cond.i.i, label %exit

exit: ; preds = %for.body.i.i
  unreachable
}
