; RUN: llvm-as < %s | opt -lowersetjmp -disable-output

	%struct.jmpenv = type { i32, i8 }

declare void @Perl_sv_setpv()

declare i32 @llvm.setjmp(i32*)

define void @perl_call_sv() {
	call void @Perl_sv_setpv( )
	%tmp.335 = getelementptr %struct.jmpenv* null, i64 0, i32 0		; <i32*> [#uses=1]
	%tmp.336 = call i32 @llvm.setjmp( i32* null )		; <i32> [#uses=1]
	store i32 %tmp.336, i32* %tmp.335
	ret void
}

