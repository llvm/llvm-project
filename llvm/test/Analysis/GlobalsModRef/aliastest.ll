; RUN: opt < %s -aa-pipeline=basic-aa,globals-aa -passes="require<globals-aa>,function(gvn)" -S -enable-unsafe-globalsmodref-alias-results | FileCheck %s
;
; Note that this test relies on an unsafe feature of GlobalsModRef. While this
; test is correct and safe, GMR's technique for handling this isn't generally.

@X = internal global i32 4		; <ptr> [#uses=1]

define i32 @test(ptr %P) {
; CHECK:      @test
; CHECK-NEXT: store i32 7, ptr %P
; CHECK-NEXT: store i32 12, ptr @X
; CHECK-NEXT: ret i32 7
	store i32 7, ptr %P
	store i32 12, ptr @X
	%V = load i32, ptr %P		; <i32> [#uses=1]
	ret i32 %V
}
