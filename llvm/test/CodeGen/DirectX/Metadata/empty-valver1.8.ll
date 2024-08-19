; RUN: opt -S -passes="print<dxil-metadata>" -disable-output %s 2>&1 | FileCheck %s --check-prefix=ANALYSIS
target datalayout = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64"
target triple = "dxilv1.6-unknown-shadermodel6.6-compute"

; ANALYSIS: Shader Model Version : 6.6
; ANALYSIS-NEXT: DXIL Version : 1.6
; ANALYSIS-NEXT: Shader Stage : compute
; ANALYSIS-NEXT: Validator Version : 1.8
; ANALYSIS-EMPTY:

; Function Attrs: nounwind memory(none)
define void @main() local_unnamed_addr #0 {
entry:
  ret void
}

attributes #0 = { nounwind memory(none) }

!dx.valver = !{!0}
!dx.shaderModel = !{!2}
!dx.version = !{!3}
!dx.entryPoints = !{!4}
!llvm.module.flags = !{!7, !8}

!0 = !{i32 1, i32 8}
!2 = !{!"cs", i32 6, i32 6}
!3 = !{i32 1, i32 6}
!4 = !{ptr @main, !"main", null, null, !5}
!5 = !{i32 4, !6}
!6 = !{i32 1, i32 1, i32 1}
!7 = !{i32 1, !"wchar_size", i32 4}
!8 = !{i32 2, !"frame-pointer", i32 2}
