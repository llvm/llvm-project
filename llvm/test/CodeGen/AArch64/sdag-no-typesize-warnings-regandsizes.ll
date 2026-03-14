; RUN: llc -mtriple=aarch64-unknown-linux-gnu -mattr=+sve < %s | FileCheck %s

; If this check fails please read
; clang/test/CodeGen/aarch64-sve-intrinsics/README for instructions on
; how to resolve it.

; CHECK-LABEL: do_something:
define <vscale x 2 x double> @do_something(<vscale x 2 x double> %vx) {
entry:
  call void @llvm.dbg.value(metadata <vscale x 2 x double> %vx, metadata !3, metadata !DIExpression()), !dbg !5
  %0 = tail call <vscale x 2 x double> @f(<vscale x 2 x double> %vx)
  ret <vscale x 2 x double> %0
}

declare <vscale x 2 x double> @f(<vscale x 2 x double>)

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1)
!1 = !DIFile(filename: "file.c", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DILocalVariable(scope: !4)
!4 = distinct !DISubprogram(unit: !0)
!5 = !DILocation(scope: !4)
