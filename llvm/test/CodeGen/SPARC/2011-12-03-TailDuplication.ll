; RUN: llc -march=sparc <%s

define void @foo(i32 %a) nounwind {
entry:
  br i1 undef, label %return, label %else.0

else.0:
  br i1 undef, label %if.end.0, label %return

if.end.0:
  br i1 undef, label %if.then.1, label %else.1

else.1:
  %0 = bitcast i8* undef to i8**
  br label %else.1.2

if.then.1:
  br i1 undef, label %return, label %return

else.1.2:
  br i1 undef, label %return, label %return

return:
  ret void
}
