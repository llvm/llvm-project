; RUN: split-file %s %t
; RUN: cat %t/shader.ll %t/v10.ll > %t/10.ll
; RUN: cat %t/shader.ll %t/v14.ll > %t/14.ll
; RUN: opt -S -dxil-translate-metadata %t/10.ll | FileCheck %s --check-prefix=MD10
; RUN: opt -S -dxil-translate-metadata %t/14.ll | FileCheck %s --check-prefix=MD14

; Validator 1.0 omits view ID state and signature metadata from the entry point.
; Validator 1.4 emits view ID state and translated input/output signatures.
; MD10-NOT: !dx.viewIdState
; MD10-NOT: !dx.semantic.signatures
; MD10: !dx.entryPoints =
; MD10-NOT: !dx.viewIdState
; MD10-NOT: !dx.semantic.signatures
; MD14: !dx.viewIdState =
; MD14: !{i32 0, !"A", i8 9, i8 0, !{{[0-9]+}}, i8 0, i32 1, i8 1, i32 0, i8 0, null}
; MD14: !{i32 0, !"B", i8 9, i8 0, !{{[0-9]+}}, i8 0, i32 1, i8 1, i32 0, i8 0, null}

;--- shader.ll
target triple = "dxil-pc-shadermodel6.0-vertex"
define void @main() #0 {
  %x = call float @llvm.dx.load.input.f32(i32 0, i32 0, i8 0, i32 poison)
  call void @llvm.dx.store.output.f32(i32 0, i32 0, i8 0, float %x)
  ret void
}
attributes #0 = { "hlsl.shader"="vertex" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{!3}
!2 = !{!4}
!3 = !{i32 0, !"A", i32 9, i32 0, !5, i32 0, i32 1, i8 1, i32 0, i8 0, i8 15, i8 15, i32 0}
!4 = !{i32 0, !"B", i32 9, i32 0, !5, i32 0, i32 1, i8 1, i32 0, i8 0, i8 15, i8 15, i32 0}
!5 = !{i32 0}

;--- v10.ll
!dx.valver = !{!99}
!99 = !{i32 1, i32 0}
;--- v14.ll
!dx.valver = !{!99}
!99 = !{i32 1, i32 4}
