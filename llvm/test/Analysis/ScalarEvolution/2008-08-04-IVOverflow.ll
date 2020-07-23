; RUN: opt < %s -analyze -enable-new-pm=0 -scalar-evolution -scalar-evolution-max-iterations=0 | FileCheck %s
; RUN: opt < %s -disable-output "-passes=print<scalar-evolution>" -scalar-evolution-max-iterations=0 2>&1 | FileCheck %s
; PR2621

define i32 @a() nounwind  {
entry:
	br label %bb1

bb:
	trunc i32 %i.0 to i16
	add i16 %0, %x16.0
	add i32 %i.0, 1
	br label %bb1

bb1:
	%i.0 = phi i32 [ 0, %entry ], [ %2, %bb ]
	%x16.0 = phi i16 [ 0, %entry ], [ %1, %bb ]
	icmp ult i32 %i.0, 888888
	br i1 %3, label %bb, label %bb2

bb2:
	zext i16 %x16.0 to i32
	ret i32 %4
}

; CHECK: Exits: 20028

