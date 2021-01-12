; RUN: opt -instcombine -S < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; This test is defending against a TypeSize message raised in the method
; `valueCoversEntireFragment` in Local.cpp because of an implicit cast from
; `TypeSize` to `uint64_t`. This particular TypeSize message only occurred when
; debug info was available.

; If this check fails please read
; clang/test/CodeGen/aarch64-sve-intrinsics/README for instructions on
; how to resolve it.
; This test must not produce any warnings. Prior to this test being introduced,
; it produced a warning containing the text "TypeSize is not scalable".
; WARN-NOT: warning:

; CHECK-LABEL: @debug_local_scalable(
define <vscale x 2 x double> @debug_local_scalable(<vscale x 2 x double> %tostore) {
  %vx = alloca <vscale x 2 x double>, align 16
  call void @llvm.dbg.declare(metadata <vscale x 2 x double>* %vx, metadata !3, metadata !DIExpression()), !dbg !5
  store <vscale x 2 x double> %tostore, <vscale x 2 x double>* %vx, align 16
  %ret = call <vscale x 2 x double> @f(<vscale x 2 x double>* %vx)
  ret <vscale x 2 x double> %ret
}

declare <vscale x 2 x double> @f(<vscale x 2 x double>*)

declare void @llvm.dbg.declare(metadata, metadata, metadata)

!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1)
!1 = !DIFile(filename: "/tmp/test.c", directory: "/tmp/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DILocalVariable(scope: !4)
!4 = distinct !DISubprogram(unit: !0)
!5 = !DILocation(scope: !4)
