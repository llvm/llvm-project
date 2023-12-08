; RUN: llc < %s -mtriple=x86_64-apple-darwin10 | FileCheck %s
; Quote should be outside of private prefix.
; rdar://6855766x

; CHECK: "l__ZZ20-[Example1 whatever]E4C.91"

	%struct.A = type { ptr, i32 }
@"_ZZ20-[Example1 whatever]E4C.91" = private constant %struct.A { ptr null, i32 1 }		; <ptr> [#uses=1]

define internal ptr @"\01-[Example1 whatever]"() nounwind optsize ssp {
entry:
	%0 = getelementptr %struct.A, ptr @"_ZZ20-[Example1 whatever]E4C.91", i64 0, i32 0		; <ptr> [#uses=1]
	%1 = load ptr, ptr %0, align 8		; <ptr> [#uses=1]
	ret ptr %1
}
