; RUN: llc < %s -mtriple=i386-apple-darwin9

	%struct._Unwind_Context = type { [18 x ptr], ptr, ptr, ptr, %struct.dwarf_eh_bases, i32, i32, i32, [18 x i8] }
	%struct._Unwind_Exception = type { i64, ptr, i32, i32, [3 x i32] }
	%struct.dwarf_eh_bases = type { ptr, ptr, ptr }

declare fastcc void @uw_init_context_1(ptr, ptr, ptr)

declare ptr @llvm.eh.dwarf.cfa(i32) nounwind

define hidden void @_Unwind_Resume(ptr %exc) noreturn noreturn {
entry:
	%0 = call ptr @llvm.eh.dwarf.cfa(i32 0)		; <ptr> [#uses=1]
	call fastcc void @uw_init_context_1(ptr null, ptr %0, ptr null)
	unreachable
}
