; RUN: llc --mtriple=m68k -filetype=obj %s -o %t
; RUN: llvm-dwarfdump -v %t | FileCheck %s

; CHECK-LABEL: DW_TAG_subprogram
; CHECK: DW_AT_calling_convention [DW_FORM_data1]        (DW_CC_LLVM_M68kRTD)
define m68k_rtdcc void @foo() !dbg !3 {
entry:
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 17.0.0 (https://github.com/llvm/llvm-project.git)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "foo.c", directory: "/path/to/file")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "foo", scope: !4, file: !4, line: 4, type: !5, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !7)
!4 = !DIFile(filename: "./foo.c", directory: "/path/to/file")
!5 = !DISubroutineType(cc: DW_CC_LLVM_M68kRTD, types: !6)
!6 = !{null}
!7 = !{}
