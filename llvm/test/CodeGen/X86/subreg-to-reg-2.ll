; RUN: llc < %s -mtriple=x86_64-apple-darwin | grep movl
; rdar://6707985

	%XXOO = type { %"struct.XXC::XXCC", i8*, %"struct.XXC::XXOO::$_71" }
	%XXValue = type opaque
	%"struct.XXC::ArrayStorage" = type { i32, i32, i32, i8*, i8*, [1 x %XXValue*] }
	%"struct.XXC::XXArray" = type { %XXOO, i32, %"struct.XXC::ArrayStorage"* }
	%"struct.XXC::XXCC" = type { i32 (...)**, i8* }
	%"struct.XXC::XXOO::$_71" = type { [2 x %XXValue*] }

define internal fastcc %XXValue* @t(i64* %out, %"struct.XXC::ArrayStorage"* %tmp9) nounwind {
prologue:
	%array = load %XXValue** inttoptr (i64 11111111 to %XXValue**)		; <%XXValue*> [#uses=0]
	%index = load %XXValue** inttoptr (i64 22222222 to %XXValue**)		; <%XXValue*> [#uses=1]
	%tmp = ptrtoint %XXValue* %index to i64		; <i64> [#uses=2]
	store i64 %tmp, i64* %out
	%tmp6 = trunc i64 %tmp to i32		; <i32> [#uses=1]
	br label %bb5

bb5:		; preds = %prologue
	%tmp10 = zext i32 %tmp6 to i64		; <i64> [#uses=1]
	%tmp11 = getelementptr %"struct.XXC::ArrayStorage"* %tmp9, i64 0, i32 5, i64 %tmp10		; <%XXValue**> [#uses=1]
	%tmp12 = load %XXValue** %tmp11, align 8		; <%XXValue*> [#uses=1]
	ret %XXValue* %tmp12
}
