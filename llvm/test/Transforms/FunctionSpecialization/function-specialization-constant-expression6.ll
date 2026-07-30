; RUN: opt -passes="ipsccp<func-spec>" -force-specialization -S < %s
; Check that we don't crash when SwitchInst Constant is not ConstantInt.

@S = external constant [1 x i8]

define i1 @foo() {
entry:
  %tmp = call i32 @bar(ptr @S)
  ret i1 0
}

define i32 @bar(ptr %arg) {
entry:
  %magicptr = ptrtoint ptr %arg to i64
  switch i64 %magicptr, label %bb2 [
    i64 0, label %bb1
  ]
bb1:
  ret i32 0
bb2:
  ret i32 1
}
