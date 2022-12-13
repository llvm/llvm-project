; RUN: opt < %s -aa-pipeline=basic-aa -passes=gvn,instcombine -S | FileCheck %s
; PR1109

target datalayout = "e-p:32:32"
target triple = "i686-apple-darwin8"
	%struct.CONSTRAINT = type { i32, i32, i32, i32 }
	%struct.FILE_POS = type { i8, i8, i16, i32 }
	%struct.FIRST_UNION = type { %struct.FILE_POS }
	%struct.FOURTH_UNION = type { %struct.CONSTRAINT }
	%struct.GAP = type { i8, i8, i16 }
	%struct.LIST = type { ptr, ptr }
	%struct.SECOND_UNION = type { { i16, i8, i8 } }
	%struct.STYLE = type { { %struct.GAP }, { %struct.GAP }, i16, i16, i16, i8, i8 }
	%struct.THIRD_UNION = type { { [2 x i32], [2 x i32] } }
	%struct.closure_type = type { [2 x %struct.LIST], %struct.FIRST_UNION, %struct.SECOND_UNION, %struct.THIRD_UNION, %struct.FOURTH_UNION, ptr, { ptr } }
	%struct.head_type = type { [2 x %struct.LIST], %struct.FIRST_UNION, %struct.SECOND_UNION, %struct.THIRD_UNION, %struct.FOURTH_UNION, ptr, { ptr }, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32 }
	%struct.rec = type { %struct.head_type }

; CHECK: define i32 @test
; CHECK:   %Z = sub i32 %A, %Q
; CHECK:   ret i32 %Z

define i32 @test(ptr %tmp18169) {
	%tmp18174 = getelementptr %struct.closure_type, ptr %tmp18169, i32 0, i32 4, i32 0, i32 0		; <ptr> [#uses=2]
	%A = load i32, ptr %tmp18174		; <i32> [#uses=1]

        %tmp18272 = getelementptr %struct.STYLE, ptr %tmp18174, i32 0, i32 0, i32 0, i32 2          ; <ptr> [#uses=1]
        store i16 123, ptr %tmp18272

	%Q = load i32, ptr %tmp18174		; <i32> [#uses=1]
	%Z = sub i32 %A, %Q		; <i32> [#uses=1]
	ret i32 %Z
}
