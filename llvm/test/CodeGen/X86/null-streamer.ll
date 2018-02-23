; Check the MCNullStreamer operates correctly, at least on a minimal test case.
;
; RUN: llc -filetype=null -o %t -march=x86 %s
; RUN: llc -filetype=null -o %t -mtriple=i686-cygwin %s

source_filename = "test/CodeGen/X86/null-streamer.ll"

define void @f0() {
  ret void
}

define void @f1() {
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: " ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !2, globals: !3, imports: !2)
!1 = !DIFile(filename: "file.c", directory: "")
!2 = !{}
!3 = !{!4}
!4 = !DIGlobalVariableExpression(var: !5)
!5 = !DIGlobalVariable(name: "i", linkageName: "_ZL1i", scope: null, file: !1, line: 1, type: !6, isLocal: true, isDefinition: true)
!6 = !DIBasicType(size: 32, align: 32, encoding: DW_ATE_signed)
!7 = !{i32 2, !"Dwarf Version", i32 3}
!8 = !{i32 1, !"Debug Info Version", i32 3}

