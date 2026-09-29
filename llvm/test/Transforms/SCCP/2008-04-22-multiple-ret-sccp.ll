; RUN: opt < %s -passes=sccp -S | grep "ret i32 %Z"
; rdar://5778210

declare {i32, i32} @bar(i32 %A) 

define i32 @foo() {
	%X = call {i32, i32} @bar(i32 17)
        %Y = extractvalue {i32, i32} %X, 0
	%Z = add i32 %Y, %Y
	ret i32 %Z
}
