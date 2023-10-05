; RUN: llc -debug-info-unopt -O0 --filetype=obj --cas-backend --cas=%t/cas --mccas-casid --mtriple=arm64-apple-darwin %s -o %t/debug_unopt.id
; RUN: llvm-cas-dump --cas=%t/cas --casid-file %t/debug_unopt.id | FileCheck %s
; CHECK: mc:debug_string_section llvmcas://
; CHECK-NEXT: mc:debug_string llvmcas://
; CHECK-NEXT: mc:padding      llvmcas://
; CHECK-NEXT: mc:section      llvmcas://
; CHECK: mc:debug_line_section llvmcas://
; CHECK-NEXT: mc:debug_line_unopt llvmcas://
; CHECK-NEXT: mc:padding      llvmcas://
define i32 @_Z3fooj(i32 noundef %0) #0 !dbg !10 {
  ret i32 1, !dbg !18
}
define i32 @_Z3bari(i32 noundef %0) #0 !dbg !19 {
  ret i32 1, !dbg !29
}
!llvm.module.flags = !{!2, !6}
!llvm.dbg.cu = !{!7}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!6 = !{i32 7, !"frame-pointer", i32 1}
!7 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !8, emissionKind: FullDebug, sysroot: "/Users/shubham")
!8 = !DIFile(filename: "a.cpp", directory: "/Users/shubham/Development/")
!10 = distinct !DISubprogram(type: !12, unit: !7, retainedNodes: !15)
!12 = !DISubroutineType(types: !13)
!13 = !{!14, !14}
!14 = !DIBasicType()
!15 = !{}
!18 = !DILocation(scope: !10)
!19 = distinct !DISubprogram(type: !20, unit: !7, retainedNodes: !15)
!20 = !DISubroutineType( types: !21)
!21 = !{!22, !22}
!22 = !DIBasicType()
!29 = !DILocation(scope: !19)
