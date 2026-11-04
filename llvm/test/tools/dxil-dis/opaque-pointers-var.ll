; RUN: llc --filetype=obj %s -o - | dxil-dis -o - | FileCheck %s
target triple = "dxil-unknown-shadermodel6.3-library"

; CHECK: @foo = common global i8* null
@foo = common global ptr null
