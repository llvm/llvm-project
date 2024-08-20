; RUN: opt -S -passes="print<dxil-metadata>" -disable-output %s 2>&1 | FileCheck %s --check-prefix=ANALYSIS
; Verify correctness of Shader Model version, DXIL version, Shader stage and Validator version
; obtained by Module Metadata Analysis pass from the metadata specified in the source.

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

!0 = !{i32 1, i32 8}
!2 = !{!"cs", i32 6, i32 6}
!3 = !{i32 1, i32 6}
