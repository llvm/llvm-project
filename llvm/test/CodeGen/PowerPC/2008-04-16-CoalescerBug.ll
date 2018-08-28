; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-unknown-linux-gnu | FileCheck %s
; Avoid reading memory that's already freed.

@llvm.used = appending global [1 x i8*] [ i8* bitcast (i32 (i64)* @_Z13GetSectorSizey to i8*) ], section "llvm.metadata"		; <[1 x i8*]*> [#uses=0]

; CHECK: @_Z13GetSectorSizey
; CHECK: blr

define i32 @_Z13GetSectorSizey(i64 %Base) nounwind  {
entry:
	br i1 false, label %bb, label %UnifiedReturnBlock
bb:		; preds = %entry
	%tmp10 = and i64 0, %Base		; <i64> [#uses=0]
	ret i32 0
UnifiedReturnBlock:		; preds = %entry
	ret i32 131072
}
