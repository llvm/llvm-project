; RUN: llc < %s | FileCheck %s
; PR2135

; CHECK: 24576

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i386-pc-linux-gnu"
@.str = constant [13 x i8] c"c45531m.adb\00\00"		

define void @main() nounwind {
entry:
	%tmp1 = call i1 @report__equal( i32 3, i32 3 )		
	%b.0 = select i1 %tmp1, i64 35184372088832, i64 0		
	%tmp7 = mul i64 3, %b.0		
	%tmp32 = icmp eq i64 %tmp7, 105553116266496		
	br i1 %tmp32, label %return, label %bb35
bb35:		
	call void @abort( )
	unreachable
return:		
	ret void
}

declare i1 @report__equal(i32 %x, i32 %y) nounwind

declare void @abort()
