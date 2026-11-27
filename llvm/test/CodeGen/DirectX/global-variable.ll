; RUN: llc -mtriple=dxil-pc-shadermodel6.3-library --filetype=obj -o %t.dxbc %s
; RUN: llvm-objcopy  --dump-section=DXIL=%t.bc %t.dxbc
; RUN: llvm-dis -o - %t.bc | FileCheck %s

; CHECK: @foo = common global ptr null
@foo = common global ptr null
