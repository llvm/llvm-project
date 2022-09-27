; RUN: llc --filetype=obj %s -o - | dxil-dis -o - | FileCheck %s
target triple = "dxil-unknown-shadermodel6.7-library"

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "Some Compiler", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "di-compile-unit.src", directory: "/some-path")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 2}
!4 = !{i32 2, !"Debug Info Version", i32 3}

; CHECK: !0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "Some Compiler", isOptimized: true, runtimeVersion: 0, emissionKind: 1, enums: !2)
; CHECK: !1 = !DIFile(filename: "di-compile-unit.src", directory: "/some-path")
; CHECK: !2 = !{}
; CHECK: !3 = !{i32 7, !"Dwarf Version", i32 2}
; CHECK: !4 = !{i32 2, !"Debug Info Version", i32 3}
