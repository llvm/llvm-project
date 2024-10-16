; RUN: llc -march=hexagon -mcpu=hexagonv73 -mattr=+hvxv73,+hvx-length128b < %s | FileCheck %s

; CHECK: splice.va.v64i16:
; CHECK: r{{[0-9]+}} = asl(r0,#1)
; CHECK: v0 = vror(v0,r{{[0-9]+}})
define <64 x i16> @splice.va.v64i16(<64 x i16> %a, <64 x i16> %b, i32 %c) #0 {
  %res = tail call <64 x i16> @llvm.vector.splice.va.v64i16(<64 x i16> %a, <64 x i16> %a, i32 %c)
  ret <64 x i16> %res
}

; CHECK: splice.va.v64i32:
; CHECK: r{{[0-9]+}} = asl(r0,#2)
; CHECK: v{{[0-9]+}} = valign(v0,v1,r{{[0-9]+}})
; CHECK: v{{[0-9]+}} = valign(v1,v0,r{{[0-9]+}})
; CHECK: v1:0 = vcombine(v{{[0-9]+}},v{{[0-9]+}})
define <64 x i32> @splice.va.v64i32(<64 x i32> %a, <64 x i32> %b, i32 %c) #0 {
  %res = tail call <64 x i32> @llvm.vector.splice.va.v64i32(<64 x i32> %a, <64 x i32> %a, i32 %c)
  ret <64 x i32> %res
}

; CHECK: splice.va.v32i32:
; CHECK: r{{[0-9]+}} = asl(r0,#2)
; CHECK: v0 = vror(v0,r{{[0-9]+}})
define <32 x i32> @splice.va.v32i32(<32 x i32> %a, <32 x i32> %b, i32 %c) #0 {
  %res = tail call <32 x i32> @llvm.vector.splice.va.v32i32(<32 x i32> %a, <32 x i32> %a, i32 %c)
  ret <32 x i32> %res
}

; CHECK: splice.va.v16i32:
; CHECK: r0 = asl(r0,#2)
; CHECK: r{{[0-9]+}} = #64
; CHECK: v1:0 = vshuff(v0,v0,r{{[0-9]+}})
; CHECK: v0 = valign(v0,v0,r0)
define <16 x i32> @splice.va.v16i32(<16 x i32> %a, <16 x i32> %b, i32 %c) #0 {
  %res = tail call <16 x i32> @llvm.vector.splice.va.v16i32(<16 x i32> %a, <16 x i32> %a, i32 %c)
  ret <16 x i32> %res
}

; CHECK: splice.va.v128i8:
; CHECK: v0 = vror(v0,r0)
define <128 x i8> @splice.va.v128i8(<128 x i8> %a, <128 x i8> %b, i32 %c) #0 {
  %res = tail call <128 x i8> @llvm.vector.splice.va.v128i8(<128 x i8> %a, <128 x i8> %a, i32 %c)
  ret <128 x i8> %res
}

declare <16 x i32> @llvm.vector.splice.va.v16i32(<16 x i32>, <16 x i32>, i32)
declare <32 x i32> @llvm.vector.splice.va.v32i32(<32 x i32>, <32 x i32>, i32)
declare <64 x i32> @llvm.vector.splice.va.v64i32(<64 x i32>, <64 x i32>, i32)
declare <64 x i16> @llvm.vector.splice.va.v64i16(<64 x i16>, <64 x i16>, i32)
declare <128 x i8> @llvm.vector.splice.va.v128i8(<128 x i8>, <128 x i8>, i32)

attributes #0 = { nounwind "target-cpu"="hexagonv73" "target-features"="+hvxv73,+hvx-length128b" }
