; RUN: opt -passes=instcombine -S < %s | FileCheck %s

; This test is defending against a TypeSize message raised in the method
; `valueCoversEntireFragment` in Local.cpp because of an implicit cast from
; `TypeSize` to `uint64_t`. This particular TypeSize message only occurred when
; debug info was available.

; CHECK-LABEL: @debug_local_scalable(
define <vscale x 2 x double> @debug_local_scalable(<vscale x 2 x double> %tostore) {
  %vx = alloca <vscale x 2 x double>, align 16
  call void @llvm.dbg.declare(metadata <vscale x 2 x double>* %vx, metadata !3, metadata !DIExpression()), !dbg !5
  store <vscale x 2 x double> %tostore, <vscale x 2 x double>* %vx, align 16
  %ret = call <vscale x 2 x double> @f(<vscale x 2 x double>* %vx)
  ret <vscale x 2 x double> %ret
}

declare <vscale x 2 x double> @f(<vscale x 2 x double>*)

define float @debug_scalablevec_bitcast_to_scalar() {
  %v.addr = alloca <vscale x 4 x float>, align 16
  call void @llvm.dbg.declare(metadata <vscale x 4 x float>* %v.addr, metadata !3, metadata !DIExpression()), !dbg !5
  %a = bitcast <vscale x 4 x float>* %v.addr to float*
  %b = load float, float* %a, align 16
  ret float %b
}

declare void @llvm.dbg.declare(metadata, metadata, metadata)

!llvm.module.flags = !{!2}
!llvm.dbg.cu = !{!0}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1)
!1 = !DIFile(filename: "/tmp/test.c", directory: "/tmp/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DILocalVariable(scope: !4)
!4 = distinct !DISubprogram(unit: !0)
!5 = !DILocation(scope: !4)
