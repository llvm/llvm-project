;RUN: llc -march=hexagon -filetype=obj < %s -o - | llvm-objdump -mv60 -mhvx -d - | FileCheck %s

; Should not crash! and map to vxor

target triple = "hexagon"

@g0 = common global <32 x i32> zeroinitializer, align 128

; Function Attrs: nounwind
define i32 @f0() #0 {
b0:
  %v0 = call <32 x i32> @llvm.hexagon.V6.vd0.128B()
  store <32 x i32> %v0, <32 x i32>* @g0, align 128
  ret i32 0
}
; CHECK: { v{{[0-9]}} = vxor(v{{[0-9]}},v{{[0-9]}})

; Function Attrs: nounwind
define i32 @f1(i32 %a0) #0 {
b0:
  %v0 = alloca i8, align 1
  %v1 = alloca i8, align 1
  %v2 = tail call i64 @llvm.hexagon.S2.asr.i.p.rnd.goodsyntax(i64 5, i32 0)
  %v3 = trunc i64 %v2 to i8
  store volatile i8 %v3, i8* %v0, align 1
  %v4 = tail call i64 @llvm.hexagon.S2.asr.i.p.rnd.goodsyntax(i64 4, i32 4)
  %v5 = trunc i64 %v4 to i8
  store volatile i8 %v5, i8* %v1, align 1
  %v6 = load volatile i8, i8* %v0, align 1
  %v7 = zext i8 %v6 to i32
  %v8 = load volatile i8, i8* %v1, align 1
  %v9 = zext i8 %v8 to i32
  %v10 = add nuw nsw i32 %v9, %v7
  ret i32 %v10
}
; CHECK: combine(#0,#4)
; CHECK: r{{[0-9]}}:{{[0-9]}} = asr(r{{[0-9]}}:{{[0-9]}},#3):rnd

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.S2.asr.i.p.rnd.goodsyntax(i64, i32) #1

; Function Attrs: nounwind
define i32 @f2(i32 %a0) #0 {
b0:
  %v0 = alloca i8, align 1
  %v1 = alloca i8, align 1
  %v2 = tail call i64 @llvm.hexagon.S5.vasrhrnd.goodsyntax(i64 6, i32 0)
  %v3 = trunc i64 %v2 to i8
  store volatile i8 %v3, i8* %v0, align 1
  %v4 = tail call i64 @llvm.hexagon.S5.vasrhrnd.goodsyntax(i64 4, i32 4)
  %v5 = trunc i64 %v4 to i8
  store volatile i8 %v5, i8* %v0, align 1
  %v6 = load volatile i8, i8* %v0, align 1
  %v7 = zext i8 %v6 to i32
  %v8 = load volatile i8, i8* %v1, align 1
  %v9 = zext i8 %v8 to i32
  %v10 = add nuw nsw i32 %v9, %v7
  ret i32 %v10
}
; CHECK: combine(#0,#4)
; CHECK: r{{[0-9]}}:{{[0-9]}} = vasrh(r{{[0-9]}}:{{[0-9]}},#3):raw

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.S5.vasrhrnd.goodsyntax(i64, i32) #1

; Function Attrs: nounwind
define i32 @f3(i32 %a0) #0 {
b0:
  %v0 = alloca i8, align 1
  %v1 = alloca i8, align 1
  %v2 = tail call i32 @llvm.hexagon.S5.asrhub.rnd.sat.goodsyntax(i64 0, i32 0)
  %v3 = trunc i32 %v2 to i8
  store volatile i8 %v3, i8* %v0, align 1
  %v4 = tail call i32 @llvm.hexagon.S5.asrhub.rnd.sat.goodsyntax(i64 4, i32 4)
  %v5 = trunc i32 %v4 to i8
  store volatile i8 %v5, i8* %v1, align 1
  %v6 = load volatile i8, i8* %v0, align 1
  %v7 = zext i8 %v6 to i32
  %v8 = load volatile i8, i8* %v1, align 1
  %v9 = zext i8 %v8 to i32
  %v10 = add nuw nsw i32 %v9, %v7
  ret i32 %v10
}
; CHECK: r{{[0-9]}} = vasrhub(r{{[0-9]}}:{{[0-9]}},#3):raw
; CHECK: r{{[0-9]}} = vsathub(r{{[0-9]}}:{{[0-9]}})

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.S5.asrhub.rnd.sat.goodsyntax(i64, i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vd0.128B() #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length128b" }
attributes #1 = { nounwind readnone }
