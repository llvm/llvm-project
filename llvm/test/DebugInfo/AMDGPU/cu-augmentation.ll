; RUN: llc -mcpu=gfx900 -mtriple=amdgcn-amd-amdhsa -filetype=obj -o - %s | llvm-dwarfdump --debug-info - | FileCheck %s

; CHECK: DW_TAG_compile_unit
; CHECK:   DW_AT_LLVM_augmentation   ("[llvm:v0.0]")
; CHECK: DW_TAG_subprogram
define void @func() #0 !dbg !4 {
  ret void
}

attributes #0 = { nounwind }

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, emissionKind: FullDebug)
!3 = !DIFile(filename: "file", directory: "dir")
!4 = distinct !DISubprogram(name: "func", scope: !3, file: !3, line: 0, type: !5, scopeLine: 0, unit: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{}
