; RUN: llc < %s

	%struct.exception = type { i8, i8, i32, ptr, ptr, i32, ptr }
@program_error = external global %struct.exception		; <ptr> [#uses=1]

define void @typeinfo() {
entry:
	%eh_typeid = tail call i32 @llvm.eh.typeid.for( ptr @program_error )		; <i32> [#uses=0]
	ret void
}

declare i32 @llvm.eh.typeid.for(ptr)
