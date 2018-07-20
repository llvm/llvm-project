; RUN: llc -march=hexagon -O0 < %s | FileCheck %s

; CHECK: r{{[0-9]*}} += rol(r{{[0-9]*}},#31)
; CHECK: r{{[0-9]*}} &= rol(r{{[0-9]*}},#31)
; CHECK: r{{[0-9]*}} -= rol(r{{[0-9]*}},#31)
; CHECK: r{{[0-9]*}} |= rol(r{{[0-9]*}},#31)
; CHECK: r{{[0-9]*}} ^= rol(r{{[0-9]*}},#31)

target triple = "hexagon"

@g0 = common global i32 0, align 4
@g1 = common global i32 0, align 4
@g2 = common global i32 0, align 4
@g3 = common global i32 0, align 4
@g4 = common global i32 0, align 4

; Function Attrs: nounwind
define i32 @f0() #0 {
b0:
  %v0 = alloca i32, align 4
  %v1 = alloca i32, align 4
  store i32 0, i32* %v0
  store i32 0, i32* %v1, align 4
  %v2 = call i32 @llvm.hexagon.S6.rol.i.r.acc(i32 0, i32 1, i32 31)
  store i32 %v2, i32* @g0, align 4
  %v3 = call i32 @llvm.hexagon.S6.rol.i.r.and(i32 0, i32 1, i32 31)
  store i32 %v3, i32* @g1, align 4
  %v4 = call i32 @llvm.hexagon.S6.rol.i.r.nac(i32 0, i32 1, i32 31)
  store i32 %v4, i32* @g2, align 4
  %v5 = call i32 @llvm.hexagon.S6.rol.i.r.or(i32 0, i32 1, i32 31)
  store i32 %v5, i32* @g3, align 4
  %v6 = call i32 @llvm.hexagon.S6.rol.i.r.xacc(i32 0, i32 1, i32 31)
  store i32 %v6, i32* @g4, align 4
  ret i32 0
}

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.S6.rol.i.r.acc(i32, i32, i32) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.S6.rol.i.r.and(i32, i32, i32) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.S6.rol.i.r.nac(i32, i32, i32) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.S6.rol.i.r.or(i32, i32, i32) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.S6.rol.i.r.xacc(i32, i32, i32) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" }
attributes #1 = { nounwind readnone }
