; RUN: opt < %s -instcombine -S | FileCheck %s

define i1 @test0(i39 %X, i39 %A) {
; CHECK: @test0
; CHECK: %[[V1:.*]] = shl i39 1, %A
; CHECK: %[[V2:.*]] = and i39 %[[V1]], %X
; CHECK: %[[V3:.*]] = icmp ne i39 %[[V2]], 0
; CHECK: ret i1 %[[V3]]

	%B = lshr i39 %X, %A
	%D = trunc i39 %B to i1
	ret i1 %D
}

define i1 @test1(i799 %X, i799 %A) {
; CHECK: @test1
; CHECK: %[[V1:.*]] = shl i799 1, %A
; CHECK: %[[V2:.*]] = and i799 %[[V1]], %X
; CHECK: %[[V3:.*]] = icmp ne i799 %[[V2]], 0
; CHECK: ret i1 %[[V3]]

	%B = lshr i799 %X, %A
	%D = trunc i799 %B to i1
	ret i1 %D
}
