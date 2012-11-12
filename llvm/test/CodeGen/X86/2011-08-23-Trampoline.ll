; RUN: llc < %s -march=x86
; RUN: llc < %s -march=x86-64

	%struct.FRAME.gnat__perfect_hash_generators__select_char_position__build_identical_keys_sets = type { i32, i32, void (i32, i32)*, i8 (i32, i32)* }

define fastcc i32 @gnat__perfect_hash_generators__select_char_position__build_identical_keys_sets.5146(i64 %table.0.0, i64 %table.0.1, i32 %last, i32 %pos) {
entry:
	call void @llvm.init.trampoline( i8* null, i8* bitcast (void (%struct.FRAME.gnat__perfect_hash_generators__select_char_position__build_identical_keys_sets*, i32, i32)* @gnat__perfect_hash_generators__select_char_position__build_identical_keys_sets__move.5177 to i8*), i8* null )		; <i8*> [#uses=0]
        %tramp22 = call i8* @llvm.adjust.trampoline( i8* null)
	unreachable
}

declare void @gnat__perfect_hash_generators__select_char_position__build_identical_keys_sets__move.5177(%struct.FRAME.gnat__perfect_hash_generators__select_char_position__build_identical_keys_sets* nest , i32, i32) nounwind 

declare void @llvm.init.trampoline(i8*, i8*, i8*) nounwind 
declare i8* @llvm.adjust.trampoline(i8*) nounwind 
