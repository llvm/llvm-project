; RUN: llc %s -o - -mtriple=thumbv7-apple-darwin10
; RUN: llc %s -o -

; REQUIRES: arm-registered-target

; NVPTX cannot select BlockAddress
; XFAIL: target=nvptx{{.*}}

; The behavior of this test is not well defined. On PowerPC the test may pass
; or fail depending on the order in which the test functions are processed by
; llc.
; UNSUPPORTED: target=powerpc{{.*}}

;; Reference to a label that gets deleted.
define ptr @test1() nounwind {
entry:
	ret ptr blockaddress(@test1b, %test_label)
}

define i32 @test1b() nounwind {
entry:
	ret i32 -1
test_label:
	br label %ret
ret:
	ret i32 -1
}


; Issues with referring to a label that gets RAUW'd later.
define i32 @test2a() nounwind {
entry:
        %target = bitcast ptr blockaddress(@test2b, %test_label) to ptr

        call i32 @test2b(ptr %target)

        ret i32 0
}

define i32 @test2b(ptr %target) nounwind {
entry:
        indirectbr ptr %target, [label %test_label]

test_label:
; assume some code here...
        br label %ret

ret:
        ret i32 -1
}

; Issues with a BB that gets RAUW'd to another one after references are
; generated.
define void @test3(ptr %P, ptr %Q) nounwind {
entry:
  store ptr blockaddress(@test3b, %test_label), ptr %P
  store ptr blockaddress(@test3b, %ret), ptr %Q
  ret void
}

define i32 @test3b() nounwind {
entry:
	br label %test_label
test_label:
	br label %ret
ret:
	ret i32 -1
}


; PR6673

define i64 @test4a() {
	%target = bitcast ptr blockaddress(@test4b, %usermain) to ptr
	%ret = call i64 @test4b(ptr %target)

	ret i64 %ret
}

define i64 @test4b(ptr %Code) {
entry:
	indirectbr ptr %Code, [label %usermain]
usermain:
	br label %label_line_0

label_line_0:
	br label %label_line_1

label_line_1:
	%target = ptrtoint ptr blockaddress(@test4b, %label_line_0) to i64
	ret i64 %target
}
