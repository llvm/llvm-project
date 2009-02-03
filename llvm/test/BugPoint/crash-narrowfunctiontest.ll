; Test that bugpoint can narrow down the testcase to the important function
;
; RUN: bugpoint %s -bugpoint-crashcalls -silence-passes > /dev/null

define i32 @foo() { ret i32 1 }

define i32 @test() {
	call i32 @test()
	ret i32 %1
}

define i32 @bar() { ret i32 2 }
