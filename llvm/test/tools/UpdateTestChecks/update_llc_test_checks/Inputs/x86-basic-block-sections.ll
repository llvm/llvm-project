; RUN: llc -mtriple=x86_64-linux -basic-block-sections=all < %s | FileCheck %s

define void @foo(i1 %cond) {
entry:
  br i1 %cond, label %true, label %false

true:
  call void @bar()
  br label %end

false:
  call void @baz()
  br label %end

end:
  ret void
}

declare void @bar()
declare void @baz()
