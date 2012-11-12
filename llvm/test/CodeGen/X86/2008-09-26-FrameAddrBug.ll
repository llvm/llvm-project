; RUN: llc < %s -mtriple=i386-apple-darwin9

	%struct._Unwind_Context = type { [18 x i8*], i8*, i8*, i8*, %struct.dwarf_eh_bases, i32, i32, i32, [18 x i8] }
	%struct._Unwind_Exception = type { i64, void (i32, %struct._Unwind_Exception*)*, i32, i32, [3 x i32] }
	%struct.dwarf_eh_bases = type { i8*, i8*, i8* }

declare fastcc void @uw_init_context_1(%struct._Unwind_Context*, i8*, i8*)

declare i8* @llvm.eh.dwarf.cfa(i32) nounwind

define hidden void @_Unwind_Resume(%struct._Unwind_Exception* %exc) noreturn noreturn {
entry:
	%0 = call i8* @llvm.eh.dwarf.cfa(i32 0)		; <i8*> [#uses=1]
	call fastcc void @uw_init_context_1(%struct._Unwind_Context* null, i8* %0, i8* null)
	unreachable
}
