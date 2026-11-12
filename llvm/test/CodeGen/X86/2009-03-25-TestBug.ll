; RUN: llc < %s -mtriple=i686-- | FileCheck %s
; rdar://6661955

; CHECK-NOT: and
; CHECK-NOT: shr

@hello = internal constant [7 x i8] c"hello\0A\00"
@world = internal constant [7 x i8] c"world\0A\00"

define void @func(ptr %b) nounwind {
bb1579.i.i:		; preds = %bb1514.i.i, %bb191.i.i
	%tmp176 = load i32, ptr %b, align 4
	%tmp177 = and i32 %tmp176, 2
	%tmp178 = icmp eq i32 %tmp177, 0
        br i1 %tmp178, label %hello, label %world

hello:
	%h = tail call i32 (ptr, ...) @printf( ptr @hello)
        ret void

world:
	%w = tail call i32 (ptr, ...) @printf( ptr @world)
        ret void
}

declare i32 @printf(ptr, ...) nounwind
