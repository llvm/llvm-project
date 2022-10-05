; RUN: llvm-reduce %s -o %t --delta-passes=metadata --test FileCheck --test-arg %s --test-arg --input-file
; RUN: FileCheck %s < %t --implicit-check-not="boring" --check-prefixes=CHECK,REDUCED

; Test that we can remove elements from named metadata.

!llvm.dbg.cu = !{!0, !1}
; CHECK: !llvm.module.flags =
; REDUCED-SAME: !{}
!llvm.module.flags = !{!10, !11, !12, !13, !14, !15, !16}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !2, isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug)
!1 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug)
; CHECK: !DIFile(filename: "interesting.c"
!2 = !DIFile(filename: "interesting.c", directory: "")
!3 = !DIFile(filename: "boring.c", directory: "")

!10 = !{i32 7, !"Dwarf Version", i32 5}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{i32 1, !"wchar_size", i32 4}
!13 = !{i32 8, !"PIC Level", i32 2}
!14 = !{i32 7, !"PIE Level", i32 2}
!15 = !{i32 7, !"uwtable", i32 2}
!16 = !{i32 7, !"frame-pointer", i32 2}
