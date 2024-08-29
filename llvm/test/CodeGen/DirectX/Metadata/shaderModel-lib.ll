; RUN: opt -S -dxil-translate-metadata %s | FileCheck %s
; RUN: opt -S -passes="print<dxil-metadata>" -disable-output %s 2>&1 | FileCheck %s --check-prefix=ANALYSIS
target triple = "dxil-pc-shadermodel6.3-library"

; CHECK: !dx.shaderModel = !{![[SM:[0-9]+]]}
; CHECK: ![[SM]] = !{!"lib", i32 6, i32 3}

; ANALYSIS: Shader Model Version : 6.3
; ANALYSIS-NEXT: DXIL Version : 1.3
; ANALYSIS-NEXT: Shader Stage : library
; ANALYSIS-NEXT: Validator Version : 0
; ANALYSIS-EMPTY:
