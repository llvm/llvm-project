; RUN: llc %s -enable-precise --filetype=asm -o - < %s 2>&1 | FileCheck %s --check-prefixes=ENABLED
; RUN: llc %s --filetype=asm -o - < %s 2>&1 | FileCheck %s --check-prefixes=DISABLED



target triple = "dxil-pc-shadermodel6.6-compute"

; ENABLED: call float @dx.op.unary.f32(i32 12, float %conv)
; ENABLED-SAME: !dx.precise ![[SM:[0-9]+]]
; ENABLED: ![[SM]] = !{i32 1}

; DISABLED-NOT: !dx.precise ![[SM:[0-9]+]]
define void @main(float %conv) {
entry:
  %1 = call float @llvm.cos.f32(float %conv)
  ret void
}
