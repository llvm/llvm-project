; RUN: llc -march=hexagon -O0 < %s | FileCheck %s
; r2 = round(r1:0):sat
; r3 = cmpyiwh(r1:0, r2):<<1:rnd:sat
; r0 = cmpyiwh(r1:0, r2*):<<1:rnd:sat

; CHECK: round(r{{[0-9]*}}:{{[0-9]*}}):sat
; CHECK: cmpyiwh(r{{[0-9]*}}:{{[0-9]*}},r{{[0-9]*}}):<<1:rnd:sat
; CHECK: cmpyrwh(r{{[0-9]*}}:{{[0-9]*}},r{{[0-9]*}}*):<<1:rnd:sat
; CHECK: cmpyiwh(r{{[0-9]*}}:{{[0-9]*}},r{{[0-9]*}}*):<<1:rnd:sat

target triple = "hexagon"

@g0 = global i32 0, align 4
@g1 = global i32 0, align 4
@g2 = global i32 0, align 4

; Function Attrs: nounwind
define i32 @f0() #0 {
b0:
  %v0 = alloca i32, align 4
  %v1 = alloca i32, align 4
  store i32 0, ptr %v0
  store i32 0, ptr %v1, align 4
  %v2 = call i32 @llvm.hexagon.A2.roundsat(i64 1)
  store i32 %v2, ptr @g1, align 4
  %v3 = call i32 @llvm.hexagon.M4.cmpyi.wh(i64 -2147483648, i32 -2147483648)
  store i32 %v3, ptr @g0, align 4
  %v4 = call i32 @llvm.hexagon.M4.cmpyr.whc(i64 2147483647, i32 2147483647)
  store i32 %v4, ptr @g2, align 4
  %v5 = call i32 @llvm.hexagon.M4.cmpyi.whc(i64 -2147483648, i32 -2147483648)
  ret i32 %v5
}

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.A2.roundsat(i64) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.M4.cmpyi.wh(i64, i32) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.M4.cmpyr.whc(i64, i32) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.M4.cmpyi.whc(i64, i32) #1

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
attributes #1 = { nounwind readnone }
