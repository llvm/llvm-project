; RUN: llc %s -o -

;; Reference to a label that gets deleted.
define i8* @test1() nounwind {
entry:
	ret i8* blockaddress(@test1b, %test_label)
}

define i32 @test1b() nounwind {
entry:
	ret i32 -1
test_label:
	br label %ret
ret:
	ret i32 -1
}


;; Issues with referring to a label that gets RAUW'd later.
define i32 @test2a() nounwind {
entry:
        %target = bitcast i8* blockaddress(@test2b, %test_label) to i8*

        call i32 @test2b(i8* %target)

        ret i32 0
}

define i32 @test2b(i8* %target) nounwind {
entry:
        indirectbr i8* %target, [label %test_label]

test_label:
; assume some code here...
        br label %ret

ret:
        ret i32 -1
}

; Issues with a BB that gets RAUW'd to another one after references are
; generated.
define void @test3(i8** %P, i8** %Q) nounwind {
entry:
  store i8* blockaddress(@test3b, %test_label), i8** %P
  store i8* blockaddress(@test3b, %ret), i8** %Q
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
	%target = bitcast i8* blockaddress(@test4b, %usermain) to i8*
	%ret = call i64 @test4b(i8* %target)

	ret i64 %ret
}

define i64 @test4b(i8* %Code) {
entry:
	indirectbr i8* %Code, [label %usermain]
usermain:
	br label %label_line_0

label_line_0:
	br label %label_line_1

label_line_1:
	%target = ptrtoint i8* blockaddress(@test4b, %label_line_0) to i64
	ret i64 %target
}
