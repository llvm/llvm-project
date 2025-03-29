; RUN: llvm-as < %s | llvm-bcanalyzer -dump | FileCheck %s

; CHECK: Block ID {{.*}} (TYPE_BLOCK_ID)
; CHECK: BFLOAT
; CHECK: TOKEN
; CHECK: HALF
; CHECK: Block ID

define half @test_half(half %x, half %y) {
  %a = fadd half %x, %y
  ret half %a
}

define bfloat @test_bfloat(i16 %x) {
  %a = bitcast i16 %x to bfloat
  ret bfloat %a
}

declare void @llvm.token(token)
