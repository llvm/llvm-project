; RUN: llc < %s
; XFAIL: sparc-sun-solaris2
; PR1308
; PR1557

define i32 @stuff(i32, ...) {
        %foo = alloca i8*
        %bar = alloca i32*
        %A = call i32 asm sideeffect "inline asm $0 $2 $3 $4", "=r,0,i,m,m"( i32 0, i32 1, i8** %foo, i32** %bar )
        ret i32 %A
}
