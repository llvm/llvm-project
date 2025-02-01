; RUN: not opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s 2>&1 | FileCheck %s

; DXIL operation UAddc only supports i32. Other integer types are unsupported.
; CHECK: in function uaddc_i16
; CHECK-SAME: Cannot create UAddc operation: Invalid overload type

define noundef i16 @uaddc_i16(i16 noundef %a, i16 noundef %b) {
  %uaddc = call { i16, i1 } @llvm.uadd.with.overflow.i16(i16 %a, i16 %b)
  %carry = extractvalue { i16, i1 } %uaddc, 1
  %sum = extractvalue { i16, i1 } %uaddc, 0
  %carry_zext = zext i1 %carry to i16
  %result = add i16 %sum, %carry_zext
  ret i16 %result
}

declare { i16, i1 } @llvm.uadd.with.overflow.i16(i16, i16)

