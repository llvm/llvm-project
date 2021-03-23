; RUN: %lli -jit-kind=mcjit -relocation-model=pic -code-model=small %s > /dev/null
; RUN: %lli -relocation-model=pic -code-model=small %s > /dev/null
; XFAIL: mips-, mipsel-, aarch64, arm, i686, i386

@count = global i32 1, align 4

define i32 @main() nounwind uwtable {
entry:
  %retval = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 0, i32* %retval
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %0, 49
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load i32, i32* @count, align 4
  %inc = add nsw i32 %1, 1
  store i32 %inc, i32* @count, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %2 = load i32, i32* %i, align 4
  %inc1 = add nsw i32 %2, 1
  store i32 %inc1, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %3 = load i32, i32* @count, align 4
  %sub = sub nsw i32 %3, 50
  ret i32 %sub
}
