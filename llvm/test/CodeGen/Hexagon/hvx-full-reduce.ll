; RUN: llc -mtriple=hexagon < %s | FileCheck %s

define i32 @full_reduce_i32_128i8_uu(<128 x i8> %x, <128 x i8> %y) #0 {
; CHECK-LABEL: full_reduce_i32_128i8_uu:
; CHECK: [[A:v[0-9]+]] = vxor([[Z:v[0-9]+]],[[Z]])
; CHECK: [[A]].uw += vrmpy(v0.ub,v1.ub)
; CHECK: valign
; CHECK: vadd
; CHECK: valign
; CHECK: vadd
; CHECK: valign
; CHECK: vadd
; CHECK: valign
; CHECK: vadd
; CHECK: valign
; CHECK: vadd
; CHECK: vextract
  %x.wide = zext <128 x i8> %x to <128 x i32>
  %y.wide = zext <128 x i8> %y to <128 x i32>
  %m = mul nuw nsw <128 x i32> %x.wide, %y.wide
  %reduce = tail call i32 @llvm.vector.reduce.add.v128i32(<128 x i32> %m)
  ret i32 %reduce
}

define i32 @full_reduce_i32_128i8_su(<128 x i8> %x, <128 x i8> %y) #0 {
; CHECK-LABEL: full_reduce_i32_128i8_su:
; CHECK: [[A:v[0-9]+]] = vxor([[Z:v[0-9]+]],[[Z]])
; CHECK: [[A]].w += vrmpy(v1.ub,v0.b)
; CHECK: valign
; CHECK: vadd
; CHECK: valign
; CHECK: vadd
; CHECK: valign
; CHECK: vadd
; CHECK: valign
; CHECK: vadd
; CHECK: valign
; CHECK: vadd
; CHECK: vextract
  %x.wide = sext <128 x i8> %x to <128 x i32>
  %y.wide = zext <128 x i8> %y to <128 x i32>
  %m = mul nuw nsw <128 x i32> %x.wide, %y.wide
  %reduce = tail call i32 @llvm.vector.reduce.add.v128i32(<128 x i32> %m)
  ret i32 %reduce
}

define i32 @full_reduce_i32_128i8_us(<128 x i8> %x, <128 x i8> %y) #0 {
; CHECK-LABEL: full_reduce_i32_128i8_us:
; CHECK: [[A:v[0-9]+]] = vxor([[Z:v[0-9]+]],[[Z]])
; CHECK: [[A]].w += vrmpy(v0.ub,v1.b)
; CHECK: valign
; CHECK: vadd
; CHECK: valign
; CHECK: vadd
; CHECK: valign
; CHECK: vadd
; CHECK: valign
; CHECK: vadd
; CHECK: valign
; CHECK: vadd
; CHECK: vextract
  %x.wide = zext <128 x i8> %x to <128 x i32>
  %y.wide = sext <128 x i8> %y to <128 x i32>
  %m = mul nuw nsw <128 x i32> %x.wide, %y.wide
  %reduce = tail call i32 @llvm.vector.reduce.add.v128i32(<128 x i32> %m)
  ret i32 %reduce
}

define i32 @full_reduce_i32_128i8_ss(<128 x i8> %x, <128 x i8> %y) #0 {
; CHECK-LABEL: full_reduce_i32_128i8_ss:
; CHECK: [[A:v[0-9]+]] = vxor([[Z:v[0-9]+]],[[Z]])
; CHECK: [[A]].w += vrmpy(v0.b,v1.b)
; CHECK: valign
; CHECK: vadd
; CHECK: valign
; CHECK: vadd
; CHECK: valign
; CHECK: vadd
; CHECK: valign
; CHECK: vadd
; CHECK: valign
; CHECK: vadd
; CHECK: vextract
  %x.wide = sext <128 x i8> %x to <128 x i32>
  %y.wide = sext <128 x i8> %y to <128 x i32>
  %m = mul nuw nsw <128 x i32> %x.wide, %y.wide
  %reduce = tail call i32 @llvm.vector.reduce.add.v128i32(<128 x i32> %m)
  ret i32 %reduce
}

;; Double-vector input.

define i32 @full_reduce_i32_256i8(<256 x i8> %x, <256 x i8> %y) #0 {
; CHECK-LABEL: full_reduce_i32_256i8:
; CHECK: vrmpy
; CHECK: vrmpy
; CHECK: vadd
; CHECK: valign
; CHECK: vadd
; CHECK: valign
; CHECK: vadd
; CHECK: valign
; CHECK: vadd
; CHECK: valign
; CHECK: vadd
; CHECK: valign
; CHECK: vadd
  %x.wide = zext <256 x i8> %x to <256 x i32>
  %y.wide = zext <256 x i8> %y to <256 x i32>
  %m = mul nuw nsw <256 x i32> %x.wide, %y.wide
  %reduce = tail call i32 @llvm.vector.reduce.add.v256i32(<256 x i32> %m)
  ret i32 %reduce
}

;; Maximum handled vector size.

define i32 @full_reduce_i32_1024i8(<1024 x i8> %x, <1024 x i8> %y) #0 {
; CHECK-LABEL: full_reduce_i32_1024i8:
; CHECK: vrmpy
; CHECK: vrmpy
; CHECK: vrmpy
; CHECK: vrmpy
; CHECK: vrmpy
; CHECK: vrmpy
; CHECK: vrmpy
; CHECK: vrmpy
; CHECK: vadd
; CHECK: valign
; CHECK: vadd
; CHECK: valign
; CHECK: vadd
; CHECK: valign
; CHECK: vadd
; CHECK: valign
; CHECK: vadd
; CHECK: valign
; CHECK: vadd
  %x.wide = zext <1024 x i8> %x to <1024 x i32>
  %y.wide = zext <1024 x i8> %y to <1024 x i32>
  %m = mul nuw nsw <1024 x i32> %x.wide, %y.wide
  %reduce = tail call i32 @llvm.vector.reduce.add.v1024i32(<1024 x i32> %m)
  ret i32 %reduce
}

attributes #0 = { nounwind readnone "target-cpu"="hexagonv68" "target-features"="+hvx,+hvx-length128b" }
