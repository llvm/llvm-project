; We use llc for this test so that we don't abort after the first error.
; RUN: not llc %s -o /dev/null 2>&1 | FileCheck %s

target triple = "dxil-pc-shadermodel6.3-library"

; DXIL operation UAddc only supports i32. Other integer types are unsupported.
; CHECK: error:
; CHECK-SAME: in function uaddc_i16
; CHECK-SAME: Cannot create UAddc operation: Invalid overload type

define noundef i16 @uaddc_i16(i16 noundef %a, i16 noundef %b) "hlsl.export" {
  %uaddc = call { i16, i1 } @llvm.uadd.with.overflow.i16(i16 %a, i16 %b)
  %carry = extractvalue { i16, i1 } %uaddc, 1
  %sum = extractvalue { i16, i1 } %uaddc, 0
  %carry_zext = zext i1 %carry to i16
  %result = add i16 %sum, %carry_zext
  ret i16 %result
}

; CHECK: error:
; CHECK-SAME: in function uaddc_return
; CHECK-SAME: DXIL ops that return structs may only be used by insert- and extractvalue

define noundef { i32, i1 } @uaddc_return(i32 noundef %a, i32 noundef %b) "hlsl.export" {
  %uaddc = call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 %a, i32 %b)
  ret { i32, i1 } %uaddc
}

declare { i16, i1 } @llvm.uadd.with.overflow.i16(i16, i16)

