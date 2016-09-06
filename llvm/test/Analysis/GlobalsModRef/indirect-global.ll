; RUN: opt < %s -basicaa -globals-aa -gvn -instcombine -S -enable-unsafe-globalsmodref-alias-results | FileCheck %s
; RUN: opt < %s -aa-pipeline=basic-aa,globals-aa -passes="require<globals-aa>,function(gvn,instcombine)" -S -enable-unsafe-globalsmodref-alias-results | FileCheck %s
;
; Note that this test relies on an unsafe feature of GlobalsModRef. While this
; test is correct and safe, GMR's technique for handling this isn't generally.

@G = internal global i32* null		; <i32**> [#uses=3]

declare i8* @malloc(i32)
define void @test() {
	%a = call i8* @malloc(i32 4)
        %A = bitcast i8* %a to i32*
	store i32* %A, i32** @G
	ret void
}

define i32 @test1(i32* %P) {
; CHECK: ret i32 0
	%g1 = load i32*, i32** @G		; <i32*> [#uses=2]
	%h1 = load i32, i32* %g1		; <i32> [#uses=1]
	store i32 123, i32* %P
	%g2 = load i32*, i32** @G		; <i32*> [#uses=0]
	%h2 = load i32, i32* %g1		; <i32> [#uses=1]
	%X = sub i32 %h1, %h2		; <i32> [#uses=1]
	ret i32 %X
}
