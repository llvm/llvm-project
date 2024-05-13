; REQUIRES: x86_64-linux
; RUN: llc -stop-after=twoaddressinstruction -mtriple=x86_64-- -o - %s | FileCheck %s


define dso_local double @twoaddressinstruction(i1 %c) local_unnamed_addr {
for.end:
  %0 = load i64, ptr undef, align 8
  br label %for.body14.preheader

for.body14.preheader:                             ; preds = %for.end
  br i1 %c, label %for.cond25.preheader.loopexit.unr-lcssa, label %for.body14.preheader.new

for.body14.preheader.new:                         ; preds = %for.body14.preheader
  %unroll_iter136 = and i64 %0, -4
  br label %for.body14

for.cond25.preheader.loopexit.unr-lcssa:          ; preds = %for.body14, %for.body14.preheader
  %indvars.iv127.unr = phi i64 [ 1, %for.body14.preheader ], [ %indvars.iv.next128.3, %for.body14 ]
  ret double undef

for.body14:                                       ; preds = %for.body14, %for.body14.preheader.new
  %indvars.iv127 = phi i64 [ 1, %for.body14.preheader.new ], [ %indvars.iv.next128.3, %for.body14 ]
  %niter137 = phi i64 [ %unroll_iter136, %for.body14.preheader.new ], [ %niter137.nsub.3, %for.body14 ]
  %indvars.iv.next128.3 = add nuw nsw i64 %indvars.iv127, 4
; CHECK: PSEUDO_PROBE -6878943695821059507, 9, 0, 0
  call void @llvm.pseudoprobe(i64 -6878943695821059507, i64 9, i32 0, i64 -1)
;; Check an opeq form of instruction is created.
; CHECK: %[[#REG:]]:gr64_nosp = COPY killed %[[#]]
; CHECK: %[[#REG]]:gr64_nosp = nuw ADD64ri32 %[[#REG]], 4, implicit-def dead $eflags
  %niter137.nsub.3 = add i64 %niter137, -4
  %niter137.ncmp.3 = icmp eq i64 %niter137.nsub.3, 0
  br i1 %niter137.ncmp.3, label %for.cond25.preheader.loopexit.unr-lcssa, label %for.body14
}

; Function Attrs: inaccessiblememonly nounwind willreturn
declare void @llvm.pseudoprobe(i64, i64, i32, i64) #0

attributes #0 = { inaccessiblememonly nounwind willreturn }
