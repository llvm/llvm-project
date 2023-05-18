; This used to be invalid, but now it's valid.  Ensure the verifier
; doesn't reject it.
; RUN: llvm-as %s -o /dev/null

declare void @doit(ptr inalloca(i64) %a)

define void @a() {
entry:
  %a = alloca inalloca [2 x i32]
  call void @doit(ptr inalloca(i64) %a)
  ret void
}

define void @b() {
entry:
  %a = alloca inalloca i64
  call void @doit(ptr inalloca(i64) %a)
  call void @doit(ptr inalloca(i64) %a)
  ret void
}

define void @c(i1 %cond) {
entry:
  br i1 %cond, label %if, label %else

if:
  %a = alloca inalloca i64
  br label %call

else:
  %b = alloca inalloca i64
  br label %call

call:
  %args = phi ptr [ %a, %if ], [ %b, %else ]
  call void @doit(ptr inalloca(i64) %args)
  ret void
}
