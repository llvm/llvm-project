; RUN: llc --filetype=obj %s -o - | dxil-dis  -o - | FileCheck %s
target triple = "dxil-unknown-shadermodel6.7-library"

; Make sure triple updated to dxil.
; CHECK:target triple = "dxil-ms-dx"
