; RUN: opt < %s -std-compile-opts | llc
; ModuleID = 'block12.c'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i686-apple-darwin8"

define void @bar() {
entry:
	br label %"LASM$foo"

"LASM$foo":		; preds = %entry
	call void asm sideeffect ".file \22block12.c\22", "~{dirflag},~{fpsr},~{flags}"( )
	call void asm sideeffect ".line 1", "~{dirflag},~{fpsr},~{flags}"( )
	call void asm sideeffect "int $$1", "~{dirflag},~{fpsr},~{flags},~{memory}"( )
	call void asm sideeffect ".file \22block12.c\22", "~{dirflag},~{fpsr},~{flags}"( )
	call void asm sideeffect ".line 2", "~{dirflag},~{fpsr},~{flags}"( )
	call void asm sideeffect "brl ${0:l}", "X,~{dirflag},~{fpsr},~{flags},~{memory}"( label %"LASM$foo" )
	br label %return

return:		; preds = %"LASM$foo"
	ret void
}

define void @baz() {
entry:
	call void asm sideeffect ".file \22block12.c\22", "~{dirflag},~{fpsr},~{flags}"( )
	call void asm sideeffect ".line 3", "~{dirflag},~{fpsr},~{flags}"( )
	call void asm sideeffect "brl ${0:l}", "X,~{dirflag},~{fpsr},~{flags},~{memory}"( label %"LASM$foo" )
	call void asm sideeffect ".file \22block12.c\22", "~{dirflag},~{fpsr},~{flags}"( )
	call void asm sideeffect ".line 4", "~{dirflag},~{fpsr},~{flags}"( )
	call void asm sideeffect "int $$1", "~{dirflag},~{fpsr},~{flags},~{memory}"( )
	br label %"LASM$foo"

"LASM$foo":		; preds = %entry
	call void asm sideeffect ".file \22block12.c\22", "~{dirflag},~{fpsr},~{flags}"( )
	call void asm sideeffect ".line 5", "~{dirflag},~{fpsr},~{flags}"( )
	call void asm sideeffect "int $$1", "~{dirflag},~{fpsr},~{flags},~{memory}"( )
	br label %return

return:		; preds = %"LASM$foo"
	ret void
}
