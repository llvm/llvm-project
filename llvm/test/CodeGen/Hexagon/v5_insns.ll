; RUN: llc -march=hexagon < %s | FileCheck %s

target triple = "hexagon"

; CHECK-LABEL: f0:
; CHECK: r{{[0-9]+}} = cmpyiwh(r{{[0-9]}}:{{[0-9]}},r{{[0-9]+}}*):<<1:rnd:sat
define i32 @f0(double %a0) {
b0:
  %v0 = alloca i8, align 1
  %v1 = fptosi double %a0 to i64
  %v2 = tail call i32 @llvm.hexagon.M4.cmpyi.whc(i64 %v1, i32 512)
  %v3 = trunc i32 %v2 to i8
  store volatile i8 %v3, i8* %v0, align 1
  %v4 = load volatile i8, i8* %v0, align 1
  %v5 = zext i8 %v4 to i32
  ret i32 %v5
}

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.M4.cmpyi.whc(i64, i32) #0

; CHECK-LABEL: f1:
; CHECK: r{{[0-9]+}} = cmpyrwh(r{{[0-9]}}:{{[0-9]}},r{{[0-9]+}}*):<<1:rnd:sat
define i32 @f1(double %a0) {
b0:
  %v0 = alloca i8, align 1
  %v1 = fptosi double %a0 to i64
  %v2 = tail call i32 @llvm.hexagon.M4.cmpyr.whc(i64 %v1, i32 512)
  %v3 = trunc i32 %v2 to i8
  store volatile i8 %v3, i8* %v0, align 1
  %v4 = load volatile i8, i8* %v0, align 1
  %v5 = zext i8 %v4 to i32
  ret i32 %v5
}

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.M4.cmpyr.whc(i64, i32) #0

; CHECK-LABEL: f2:
; CHECK: r{{[0-9]+}} = popcount(r{{[0-9]}}:{{[0-9]}})
define i32 @f2(double %a0) {
b0:
  %v0 = alloca i8, align 1
  %v1 = fptosi double %a0 to i64
  %v2 = tail call i32 @llvm.hexagon.S5.popcountp(i64 %v1)
  %v3 = trunc i32 %v2 to i8
  store volatile i8 %v3, i8* %v0, align 1
  %v4 = load volatile i8, i8* %v0, align 1
  %v5 = zext i8 %v4 to i32
  ret i32 %v5
}

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.S5.popcountp(i64) #0

; CHECK-LABEL: f3:
; CHECK: p{{[0-3]+}} = sfclass(r{{[0-9]}},#3)
define i32 @f3(float %a0) {
b0:
  %v0 = alloca i8, align 1
  %v1 = tail call i32 @llvm.hexagon.F2.sfclass(float %a0, i32 3)
  %v2 = trunc i32 %v1 to i8
  store volatile i8 %v2, i8* %v0, align 1
  %v3 = load volatile i8, i8* %v0, align 1
  %v4 = zext i8 %v3 to i32
  ret i32 %v4
}

; Function Attrs: readnone
declare i32 @llvm.hexagon.F2.sfclass(float, i32) #1

; CHECK-LABEL: f4:
; CHECK: r{{[0-9]+}} = vasrhub(r{{[0-9]}}:{{[0-9]}},#3):sat
define i32 @f4(float %a0) {
b0:
  %v0 = alloca i8, align 1
  %v1 = fptosi float %a0 to i64
  %v2 = tail call i32 @llvm.hexagon.S5.asrhub.sat(i64 %v1, i32 3)
  %v3 = trunc i32 %v2 to i8
  store volatile i8 %v3, i8* %v0, align 1
  %v4 = load volatile i8, i8* %v0, align 1
  %v5 = zext i8 %v4 to i32
  ret i32 %v5
}

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.S5.asrhub.sat(i64, i32) #0

attributes #0 = { nounwind readnone }
attributes #1 = { readnone }
