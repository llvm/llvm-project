; RUN: llc < %s -mtriple=i686--
; RUN: llc < %s -mtriple=x86_64--

	%struct.FRAME.gnat__perfect_hash_generators__select_char_position__build_identical_keys_sets = type { i32, i32, ptr, ptr }

define fastcc i32 @gnat__perfect_hash_generators__select_char_position__build_identical_keys_sets.5146(i64 %table.0.0, i64 %table.0.1, i32 %last, i32 %pos) {
entry:
	call void @llvm.init.trampoline( ptr null, ptr @gnat__perfect_hash_generators__select_char_position__build_identical_keys_sets__move.5177, ptr null )		; <ptr> [#uses=0]
        %tramp22 = call ptr @llvm.adjust.trampoline( ptr null)
	unreachable
}

declare void @gnat__perfect_hash_generators__select_char_position__build_identical_keys_sets__move.5177(ptr nest , i32, i32) nounwind 

declare void @llvm.init.trampoline(ptr, ptr, ptr) nounwind 
declare ptr @llvm.adjust.trampoline(ptr) nounwind 
