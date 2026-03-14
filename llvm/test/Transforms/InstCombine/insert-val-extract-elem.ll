; RUN: opt -S -passes=instcombine %s | FileCheck %s

; CHECK-LABEL: julia_2xdouble
; CHECK-NOT: insertvalue
; CHECK-NOT: extractelement
; CHECK: store <2 x double>
define void @julia_2xdouble(ptr sret([2 x double]), ptr) {
top:
  %x = load <2 x double>, ptr %1
  %x0 = extractelement <2 x double> %x, i32 0
  %i0 = insertvalue [2 x double] undef, double %x0, 0
  %x1 = extractelement <2 x double> %x, i32 1
  %i1 = insertvalue [2 x double] %i0, double %x1, 1
  store [2 x double] %i1, ptr %0, align 4
  ret void
}

; Test with two inserts to the same index
; CHECK-LABEL: julia_2xi64
; CHECK-NOT: insertvalue
; CHECK-NOT: extractelement
; CHECK: store <2 x i64>
define void @julia_2xi64(ptr sret([2 x i64]), ptr) {
top:
  %x = load <2 x i64>, ptr %1
  %x0 = extractelement <2 x i64> %x, i32 1
  %i0 = insertvalue [2 x i64] undef, i64 %x0, 0
  %x1 = extractelement <2 x i64> %x, i32 1
  %i1 = insertvalue [2 x i64] %i0, i64 %x1, 1
  %x2 = extractelement <2 x i64> %x, i32 0
  %i2 = insertvalue [2 x i64] %i1, i64 %x2, 0
  store [2 x i64] %i2, ptr %0, align 4
  ret void
}

; CHECK-LABEL: julia_4xfloat
; CHECK-NOT: insertvalue
; CHECK-NOT: extractelement
; CHECK: store <4 x float>
define void @julia_4xfloat(ptr sret([4 x float]), ptr) {
top:
  %x = load <4 x float>, ptr %1
  %x0 = extractelement <4 x float> %x, i32 0
  %i0 = insertvalue [4 x float] undef, float %x0, 0
  %x1 = extractelement <4 x float> %x, i32 1
  %i1 = insertvalue [4 x float] %i0, float %x1, 1
  %x2 = extractelement <4 x float> %x, i32 2
  %i2 = insertvalue [4 x float] %i1, float %x2, 2
  %x3 = extractelement <4 x float> %x, i32 3
  %i3 = insertvalue [4 x float] %i2, float %x3, 3
  store [4 x float] %i3, ptr %0, align 4
  ret void
}

%pseudovec = type { float, float, float, float }

; CHECK-LABEL: julia_pseudovec
; CHECK-NOT: insertvalue
; CHECK-NOT: extractelement
; CHECK: store <4 x float>
define void @julia_pseudovec(ptr sret(%pseudovec), ptr) {
top:
  %x = load <4 x float>, ptr %1
  %x0 = extractelement <4 x float> %x, i32 0
  %i0 = insertvalue %pseudovec undef, float %x0, 0
  %x1 = extractelement <4 x float> %x, i32 1
  %i1 = insertvalue %pseudovec %i0, float %x1, 1
  %x2 = extractelement <4 x float> %x, i32 2
  %i2 = insertvalue %pseudovec %i1, float %x2, 2
  %x3 = extractelement <4 x float> %x, i32 3
  %i3 = insertvalue %pseudovec %i2, float %x3, 3
  store %pseudovec %i3, ptr %0, align 4
  ret void
}
