; RUN: opt -passes='loop-mssa(simple-loop-unswitch<nontrivial;trivial>)' -verify-memoryssa -disable-output %s

@a = global i16 0
@buf = global i8 0

define void @test(i16 %0) {
entry:
  store i16 0, ptr @a, align 2
  br label %for.cond

for.cond:                                         ; preds = %lor.end, %entry
  br label %for.cond2.preheader

for.cond2.preheader:                              ; preds = %lor.end, %for.cond
  %arrayidx7 = getelementptr i8, ptr @buf, i64 0
  br label %for.cond2

for.cond2:                                        ; preds = %for.body4, %for.cond2.preheader
  switch i16 %0, label %lor.end [
    i16 -6, label %for.body4
    i16 0, label %lor.rhs
  ]

for.body4:                                        ; preds = %for.cond2
  %1 = load i8, ptr %arrayidx7, align 1
  br label %for.cond2

lor.rhs:                                          ; preds = %for.cond2
  br label %lor.end

lor.end:                                          ; preds = %lor.rhs, %for.cond2
  br i1 false, label %for.cond, label %for.cond2.preheader
}
