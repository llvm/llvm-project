; RUN: opt -passes='require<memoryssa>,adce' -o - -S -debug-pass-manager < %s 2>&1 | FileCheck %s

; ADCE should remove the dbg.declare in test1, but it should preserve
; MemorySSA since it only removed a debug instruction.

; Invalidating MemorySSA when only removing debug instructions may lead
; to different code with/without debug info present.
; In
;  https://github.com/llvm/llvm-project/issues/58285
; we saw how ADCE removed a dbg.declare, invalidated all analyses, and later
; DSE behaved in some way. Without the dbg.declare in the input ADCE kept
; analysis info, and DSE behaved differently.

; CHECK: Running analysis: MemorySSAAnalysis on test1
; CHECK: Running pass: ADCEPass on test1 (1 instruction)
; CHECK-NOT: Invalidating analysis: MemorySSAAnalysis on test1

; In test2 ADCE also removes an instruction, but since we remove a non-debug
; instruction as well we invalidate several analyses, including MemorySSA.

; CHECK: Running analysis: MemorySSAAnalysis on test2
; CHECK: Running pass: ADCEPass on test2 (2 instructions)
; CHECK: Invalidating analysis: MemorySSAAnalysis on test2
; CHECK: Running pass: PrintModulePass

; CHECK-LABEL: @test1(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i16 0

; CHECK-LABEL: @test2(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i16 0

define i16 @test1() {
entry:
  call void @llvm.dbg.declare(metadata ptr poison, metadata !4, metadata !DIExpression()), !dbg !16
  ret i16 0
}

define i16 @test2() {
entry:
  %dead = add i16 1, 2
  ret i16 0
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.declare(metadata, metadata, metadata) #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.dbg.cu = !{}
!llvm.module.flags = !{!0, !1, !2, !3}

!0 = !{i32 7, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = !{i32 1, !"wchar_size", i32 1}
!3 = !{i32 7, !"frame-pointer", i32 2}
!4 = !DILocalVariable(name: "w", scope: !5, file: !6, line: 18, type: !11)
!5 = distinct !DILexicalBlock(scope: !7, file: !6, line: 18, column: 8)
!6 = !DIFile(filename: "foo2.c", directory: "/llvm")
!7 = distinct !DISubprogram(name: "test1", scope: !6, file: !6, line: 14, type: !8, scopeLine: 14, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !10)
!8 = !DISubroutineType(types: !9)
!9 = !{}
!10 = distinct !DICompileUnit(language: DW_LANG_C99, file: !6, producer: "clang version 16.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !9, splitDebugInlining: false, nameTableKind: None)
!11 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint64_t", file: !12, line: 60, baseType: !13)
!12 = !DIFile(filename: "/include/sys/_stdint.h", directory: "")
!13 = !DIDerivedType(tag: DW_TAG_typedef, name: "__uint64_t", file: !14, line: 108, baseType: !15)
!14 = !DIFile(filename: "/include/machine/_default_types.h", directory: "")
!15 = !DIBasicType(name: "unsigned long long", size: 64, encoding: DW_ATE_unsigned)
!16 = !DILocation(line: 18, column: 8, scope: !5)