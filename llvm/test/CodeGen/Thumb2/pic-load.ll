; RUN: llc < %s -mtriple=thumbv7-apple-darwin9 -relocation-model=pic | FileCheck %s

	%struct.anon = type { void ()* }
	%struct.one_atexit_routine = type { %struct.anon, i32, i8* }
@__dso_handle = external global { }		; <{ }*> [#uses=1]
@llvm.used = appending global [1 x i8*] [i8* bitcast (i32 (void ()*)* @atexit to i8*)], section "llvm.metadata"		; <[1 x i8*]*> [#uses=0]

define hidden i32 @atexit(void ()* %func) nounwind {
entry:
; CHECK: atexit:
; CHECK: add r0, pc
	%r = alloca %struct.one_atexit_routine, align 4		; <%struct.one_atexit_routine*> [#uses=3]
	%0 = getelementptr %struct.one_atexit_routine* %r, i32 0, i32 0, i32 0		; <void ()**> [#uses=1]
	store void ()* %func, void ()** %0, align 4
	%1 = getelementptr %struct.one_atexit_routine* %r, i32 0, i32 1		; <i32*> [#uses=1]
	store i32 0, i32* %1, align 4
	%2 = call  i32 @atexit_common(%struct.one_atexit_routine* %r, i8* bitcast ({ }* @__dso_handle to i8*)) nounwind		; <i32> [#uses=1]
	ret i32 %2
}

declare i32 @atexit_common(%struct.one_atexit_routine*, i8*) nounwind
