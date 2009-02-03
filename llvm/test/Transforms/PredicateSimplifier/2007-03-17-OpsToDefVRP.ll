; RUN: llvm-as < %s | opt -predsimplify | llvm-dis | grep -v %c
define void @foo(i8* %X, i8* %Y) {
entry:
  %A = load i8* %X
  %B = load i8* %Y
  %a = icmp ult i8 %B, 10
  br i1 %a, label %cond_true, label %URB
cond_true:
  %b = icmp eq i8 %A, %B
  br i1 %b, label %cond_true2, label %URB
cond_true2:
  %c = icmp ult i8 %A, 11
  call i8 @bar(i1 %c)
  ret void
URB:
  ret void
}

declare i8 @bar(i1)
