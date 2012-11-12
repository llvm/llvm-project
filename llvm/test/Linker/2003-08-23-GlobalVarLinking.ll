; RUN: llvm-as < %s > %t.out1.bc
; RUN: echo "%%T1 = type opaque %%T2 = type opaque @S = external global { i32, %%T1* } declare void @F(%%T2*)"\
; RUN:   | llvm-as > %t.out2.bc
; RUN: llvm-link %t.out1.bc %t.out2.bc -S | not grep opaque

; After linking this testcase, there should be no opaque types left.  The two
; S's should cause the opaque type to be resolved to 'int'.
@S = global { i32, i32* } { i32 5, i32* null }		; <{ i32, i32* }*> [#uses=0]

declare void @F(i32*)
