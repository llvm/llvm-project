; RUN: llc < %s -mtriple=x86_64-apple-darwin10 | grep L__ZZ20
; Quote should be outside of private prefix.
; rdar://6855766x

	%struct.A = type { i32*, i32 }
@"_ZZ20-[Example1 whatever]E4C.91" = private constant %struct.A { i32* null, i32 1 }		; <%struct.A*> [#uses=1]

define internal i32* @"\01-[Example1 whatever]"() nounwind optsize ssp {
entry:
	%0 = getelementptr %struct.A* @"_ZZ20-[Example1 whatever]E4C.91", i64 0, i32 0		; <i32**> [#uses=1]
	%1 = load i32** %0, align 8		; <i32*> [#uses=1]
	ret i32* %1
}
