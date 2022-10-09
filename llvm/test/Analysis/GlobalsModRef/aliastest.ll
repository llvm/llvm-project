; RUN: opt < %s -aa-pipeline=basic-aa,globals-aa -passes="require<globals-aa>,function(gvn)" -S -enable-unsafe-globalsmodref-alias-results | FileCheck %s
;
; Note that this test relies on an unsafe feature of GlobalsModRef. While this
; test is correct and safe, GMR's technique for handling this isn't generally.

@X = internal global i32 4		; <i32*> [#uses=1]

define i32 @test(i32* %P) {
; CHECK:      @test
; CHECK-NEXT: store i32 7, i32* %P
; CHECK-NEXT: store i32 12, i32* @X
; CHECK-NEXT: ret i32 7
	store i32 7, i32* %P
	store i32 12, i32* @X
	%V = load i32, i32* %P		; <i32> [#uses=1]
	ret i32 %V
}
