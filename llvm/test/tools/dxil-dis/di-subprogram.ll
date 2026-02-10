; RUN: llc --filetype=obj %s -o - | dxil-dis -o - | FileCheck %s
target triple = "dxil-unknown-shadermodel6.7-library"

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

; CHECK: !0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "Some Compiler", isOptimized: true, runtimeVersion: 0, emissionKind: 1, enums: !2)
!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "Some Compiler", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
; CHECK: !1 = !DIFile(filename: "some-source", directory: "some-path")
!1 = !DIFile(filename: "some-source", directory: "some-path")
!2 = !{}

; CHECK: !3 = !{i32 2, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Dwarf Version", i32 4}
; CHECK: !4 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 2, !"Debug Info Version", i32 3}
