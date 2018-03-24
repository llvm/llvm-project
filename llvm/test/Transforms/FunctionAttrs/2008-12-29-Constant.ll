; RUN: opt < %s -basicaa -functionattrs -S | FileCheck %s
; RUN: opt < %s -aa-pipeline=basic-aa -passes=function-attrs -S | FileCheck %s

@s = external constant i8		; <i8*> [#uses=1]

; CHECK: define i8 @f() #0
define i8 @f() {
	%tmp = load i8, i8* @s		; <i8> [#uses=1]
	ret i8 %tmp
}

; CHECK: attributes #0 = { {{.*}} readnone
