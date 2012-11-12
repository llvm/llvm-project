; RUN: opt < %s -deadargelim -S > %t
; RUN: grep "define internal zeroext i32 @test1() nounwind" %t
; RUN: grep "define internal <{ i32, i32 }> @test2" %t

%Ty = type <{ i32, i32 }>

; Check if the pass doesn't modify anything that doesn't need changing. We feed
; an unused argument to each function to lure it into changing _something_ about
; the function and then changing too much.

; This checks if the return value attributes are not removed
define internal zeroext i32 @test1(i32 %DEADARG1) nounwind {
        ret i32 1
}

; This checks if the struct doesn't get non-packed
define internal <{ i32, i32 }> @test2(i32 %DEADARG1) {
        ret <{ i32, i32 }> <{ i32 1, i32 2 }>
}

; We use this external function to make sure the return values don't become dead
declare void @user(i32, <{ i32, i32 }>)

define void @caller() {
        %B = call i32 @test1(i32 1)
        %C = call <{ i32, i32 }> @test2(i32 2)
        call void @user(i32 %B, <{ i32, i32 }> %C)
        ret void
}

