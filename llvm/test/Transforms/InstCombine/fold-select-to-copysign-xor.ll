; RUN: opt -passes=instcombine -S < %s | FileCheck %s

; Test all boolean/signbit variants that should fold to copysign(y, x)
; or copysign(y, -x).

define float @copysign1(float %x, float %y) {
entry:
  %x_bits = bitcast float %x to i32
  %y_bits = bitcast float %y to i32
  %xor = xor i32 %x_bits, %y_bits
  %sign = icmp slt i32 %xor, 0
  %neg_y = fneg float %y
  %sel = select i1 %sign, float %neg_y, float %y
  ret float %sel
}

; CHECK-LABEL: define float @copysign1
; CHECK: [[CS1:%.*]] = call float @llvm.copysign.f32(float %y, float %x)
; CHECK-NEXT: ret float [[CS1]]

define float @copysign2(float %x, float %y) {
entry:
  %x_bits = bitcast float %x to i32
  %y_bits = bitcast float %y to i32
  %xor = xor i32 %x_bits, %y_bits
  %sign = icmp slt i32 %xor, 0
  %neg_y = fneg float %y
  %sel = select i1 %sign, float %neg_y, float %y
  ret float %sel
}

; CHECK-LABEL: define float @copysign2
; CHECK: [[CS2:%.*]] = call float @llvm.copysign.f32(float %y, float %x)
; CHECK-NEXT: ret float [[CS2]]

define float @copysign3(float %x, float %y) {
entry:
  %x_bits = bitcast float %x to i32
  %y_bits = bitcast float %y to i32
  %xor = xor i32 %x_bits, %y_bits
  %sign = icmp slt i32 %xor, 0
  %neg_y = fneg float %y
  %sel = select i1 %sign, float %neg_y, float %y
  ret float %sel
}

; CHECK-LABEL: define float @copysign3
; CHECK: [[CS3:%.*]] = call float @llvm.copysign.f32(float %y, float %x)
; CHECK-NEXT: ret float [[CS3]]

define float @copysign4(float %x, float %y) {
entry:
  %x_bits = bitcast float %x to i32
  %y_bits = bitcast float %y to i32
  %xor = xor i32 %x_bits, %y_bits
  %sign = icmp slt i32 %xor, 0
  %neg_y = fneg float %y
  %sel = select i1 %sign, float %y, float %neg_y
  ret float %sel
}

; CHECK-LABEL: define float @copysign4
; CHECK: [[CS4:%.*]] = call float @llvm.copysign.f32(float %y, float %0)
; CHECK-NEXT: ret float [[CS4]]

define float @copysign5(float %x, float %y) {
entry:
  %x_bits = bitcast float %x to i32
  %y_bits = bitcast float %y to i32
  %xor = xor i32 %x_bits, %y_bits
  %sign = icmp slt i32 %xor, 0
  %neg_y = fneg float %y
  %sel = select i1 %sign, float %neg_y, float %y
  ret float %sel
}

; CHECK-LABEL: define float @copysign5
; CHECK: [[CS5:%.*]] = call float @llvm.copysign.f32(float %y, float %x)
; CHECK-NEXT: ret float [[CS5]]

define float @copysign6(float %x, float %y) {
entry:
  %x_bits = bitcast float %x to i32
  %y_bits = bitcast float %y to i32
  %xor = xor i32 %x_bits, %y_bits
  %sign = icmp slt i32 %xor, 0
  %neg_y = fneg float %y
  %sel = select i1 %sign, float %y, float %neg_y
  ret float %sel
}

; CHECK-LABEL: define float @copysign6
; CHECK: [[NEGX6:%.*]] = fneg float %x
; CHECK: [[CS6:%.*]] = call float @llvm.copysign.f32(float %y, float [[NEGX6]])
; CHECK-NEXT: ret float [[CS6]]

define float @copysign7(float %x, float %y) {
entry:
  %x_bits = bitcast float %x to i32
  %y_bits = bitcast float %y to i32
  %xor = xor i32 %x_bits, %y_bits
  %sign = icmp slt i32 %xor, 0
  %neg_y = fneg float %y
  %sel = select i1 %sign, float %y, float %neg_y
  ret float %sel
}

; CHECK-LABEL: define float @copysign7
; CHECK: [[NEGX7:%.*]] = fneg float %x
; CHECK: [[CS7:%.*]] = call float @llvm.copysign.f32(float %y, float [[NEGX7]])
; CHECK-NEXT: ret float [[CS7]]

define float @copysign8(float %x, float %y) {
entry:
  %x_bits = bitcast float %x to i32
  %y_bits = bitcast float %y to i32
  %xor = xor i32 %x_bits, %y_bits
  %sign = icmp slt i32 %xor, 0
  %neg_y = fneg float %y
  %sel = select i1 %sign, float %y, float %neg_y
  ret float %sel
}

; CHECK-LABEL: define float @copysign8
; CHECK: [[NEGX8:%.*]] = fneg float %x
; CHECK: [[CS8:%.*]] = call float @llvm.copysign.f32(float %y, float [[NEGX8]])
; CHECK-NEXT: ret float [[CS8]]

define float @copysign9(float %x, float %y) {
entry:
  %x_bits = bitcast float %x to i32
  %y_bits = bitcast float %y to i32
  %xor = xor i32 %x_bits, %y_bits
  %sign = icmp slt i32 %xor, 0
  %neg_y = fneg float %y
  %sel = select i1 %sign, float %y, float %neg_y
  ret float %sel
}

; CHECK-LABEL: define float @copysign9
; CHECK: [[NEGX9:%.*]] = fneg float %x
; CHECK: [[CS9:%.*]] = call float @llvm.copysign.f32(float %y, float [[NEGX9]])
; CHECK-NEXT: ret float [[CS9]]