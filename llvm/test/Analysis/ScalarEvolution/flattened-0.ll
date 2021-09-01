; RUN: opt < %s "-passes=print<scalar-evolution>" -disable-output 2>&1 | FileCheck %s

define void @foo([7 x i8]* %a) {
; CHECK-LABEL: @foo
entry:
	br label %bb

bb:
	%idx = phi i64 [ 0, %entry ], [ %idx.incr, %bb ]
	%i = udiv i64 %idx, 7
	%j = urem i64 %idx, 7
	%a.ptr = getelementptr [7 x i8], [7 x i8]* %a, i64 %i, i64 %j
; CHECK: %a.ptr = getelementptr [7 x i8], [7 x i8]* %a, i64 %i, i64 %j
; CHECK-NEXT: -->  {%a,+,1}<nw><%bb>
	%val = load i8, i8* %a.ptr
	%idx.incr = add i64 %idx, 1
	%test = icmp ne i64 %idx.incr, 35
	br i1 %test, label %bb, label %exit

exit:
	ret void
}
