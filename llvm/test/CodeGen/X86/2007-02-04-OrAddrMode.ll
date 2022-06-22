; RUN: llc < %s -mtriple=i686-- | FileCheck %s

;; This example can't fold the or into an LEA.
define i32 @test(ptr %tmp2, i32 %tmp12) nounwind {
; CHECK-LABEL: test:
; CHECK-NOT: ret
; CHECK: orl $1, %{{.*}}
; CHECK: ret

	%tmp3 = load ptr, ptr %tmp2
	%tmp132 = shl i32 %tmp12, 2		; <i32> [#uses=1]
	%ctg2 = getelementptr i8, ptr %tmp3, i32 %tmp132		; <ptr> [#uses=1]
	%tmp6 = ptrtoint ptr %ctg2 to i32		; <i32> [#uses=1]
	%tmp14 = or i32 %tmp6, 1		; <i32> [#uses=1]
	ret i32 %tmp14
}

;; This can!
define i32 @test2(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: test2:
; CHECK-NOT: ret
; CHECK: leal 3(,%{{.*}},8)
; CHECK: ret

	%c = shl i32 %a, 3
	%d = or i32 %c, 3
	ret i32 %d
}
