; RUN: llvm-as < %s | llc -mtriple=i386-apple-darwin -mattr=+sse2 -disable-fp-elim | grep subl | grep 24

	%struct.argument_t = type { i8*, %struct.argument_t*, i32, %struct.ipc_type_t*, i32, void (...)*, void (...)*, void (...)*, void (...)*, void (...)*, i8*, i8*, i8*, i8*, i8*, i32, i32, i32, %struct.routine*, %struct.argument_t*, %struct.argument_t*, %struct.argument_t*, %struct.argument_t*, %struct.argument_t*, %struct.argument_t*, %struct.argument_t*, i32, i32, i32, i32, i32, i32 }
	%struct.ipc_type_t = type { i8*, %struct.ipc_type_t*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i8*, i8*, i32, i32, i32, i32, i32, i32, %struct.ipc_type_t*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8* }
	%struct.routine = type opaque
@"\01LC" = external constant [11 x i8]		; <[11 x i8]*> [#uses=1]

define i8* @InArgMsgField(%struct.argument_t* %arg, i8* %str) nounwind  {
entry:
	%who = alloca [20 x i8]		; <[20 x i8]*> [#uses=1]
	%who1 = getelementptr [20 x i8]* %who, i32 0, i32 0		; <i8*> [#uses=2]
	call void @llvm.memset.i32( i8* %who1, i8 0, i32 20, i32 1 )
	call void @llvm.memcpy.i32( i8* %who1, i8* getelementptr ([11 x i8]* @"\01LC", i32 0, i32 0), i32 11, i32 1 )
	unreachable
}

declare void @llvm.memset.i32(i8*, i8, i32, i32) nounwind 

declare void @llvm.memcpy.i32(i8*, i8*, i32, i32) nounwind 
