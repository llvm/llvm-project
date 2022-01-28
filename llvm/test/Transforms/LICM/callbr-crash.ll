; RUN: opt -licm -disable-output < %s

define i32 @j() {
entry:
  br label %for.cond

for.cond:                                         ; preds = %cond.true.i, %entry
  callbr void asm sideeffect "", "i,~{dirflag},~{fpsr},~{flags}"(i8* blockaddress(@j, %for.end))
          to label %cond.true.i [label %for.end]

cond.true.i:                                      ; preds = %for.cond
  %asmresult1.i.i = extractvalue { i8, i32 } zeroinitializer, 1
  br i1 undef, label %for.end, label %for.cond

for.end:                                          ; preds = %cond.true.i, %for.cond
  %asmresult1.i.i2 = phi i32 [ %asmresult1.i.i, %cond.true.i ], [ undef, %for.cond ]
  ret i32 undef
}
