; Linking these two translation units causes there to be two LLVM values in the
; symbol table with the same name and same type.  When this occurs, the symbol
; table class is DROPPING one of the values, instead of renaming it like a nice
; little symbol table.  This is causing llvm-link to die, at no fault of its
; own.

; RUN: llvm-as < %s > %t.out2.bc
; RUN: echo "%%T1 = type opaque  @GVar = external global %%T1*" | llvm-as > %t.out1.bc
; RUN: llvm-link %t.out1.bc %t.out2.bc

%T1 = type opaque
%T2 = type i32
@GVar = global i32* null		; <i32**> [#uses=0]

define void @foo(i32* %X) {
	%X.upgrd.1 = bitcast i32* %X to %T1*		; <%T1*> [#uses=0]
	ret void
}


