;; RUN: llc --mtriple=hexagon --mcpu=hexagonv81 --mattr=+hvxv81,+hvx-length128b %s -o - | FileCheck %s

define void @mul_and_sub_1(ptr readonly %A, ptr readonly %B, ptr readonly %C, ptr writeonly %D) {
entry:
  %AVec = load <32 x float>, ptr %A, align 4
  %BVec = load <32 x float>, ptr %B, align 4
  %CVec = load <32 x float>, ptr %C, align 4
  %AtBVec = fmul <32 x float> %AVec, %BVec

  %DVec = fsub <32 x float> %CVec, %AtBVec
  store <32 x float> %DVec, ptr %D, align 4
  ret void
}
;; CHECK: mul_and_sub_1
;; CHECK: vsub(v{{[0-9]+}}.sf,v{{[0-9]+}}.qf32)


define void @mul_and_sub_2(ptr readonly %A, ptr readonly %B, ptr readonly %C, ptr writeonly %D) {
entry:
  %AVec = load <32 x float>, ptr %A, align 4
  %BVec = load <32 x float>, ptr %B, align 4
  %CVec = load <32 x float>, ptr %C, align 4
  %AtBVec = fmul <32 x float> %AVec, %BVec

  %DVec = fsub <32 x float> %AtBVec, %CVec
  store <32 x float> %DVec, ptr %D, align 4
  ret void
}
;; CHECK: mul_and_sub_2
;; CHECK: vsub(v{{[0-9]+}}.qf32,v{{[0-9]+}}.sf)


define void @mul_and_sub_3(ptr readonly %A, ptr readonly %B, ptr readonly %C, ptr writeonly %D) {
entry:
  %AVec = load <64 x half>, ptr %A, align 4
  %BVec = load <64 x half>, ptr %B, align 4
  %CVec = load <64 x half>, ptr %C, align 4
  %AtBVec = fmul <64 x half> %AVec, %BVec

  %DVec = fsub <64 x half> %CVec, %AtBVec
  store <64 x half> %DVec, ptr %D, align 4
  ret void
}
;; CHECK: mul_and_sub_3
;; CHECK: vsub(v{{[0-9]+}}.hf,v{{[0-9]+}}.qf16)


define void @mul_and_sub_4(ptr readonly %A, ptr readonly %B, ptr readonly %C, ptr writeonly %D) {
entry:
  %AVec = load <64 x half>, ptr %A, align 4
  %BVec = load <64 x half>, ptr %B, align 4
  %CVec = load <64 x half>, ptr %C, align 4
  %AtBVec = fmul <64 x half> %AVec, %BVec

  %DVec = fsub <64 x half> %AtBVec, %CVec
  store <64 x half> %DVec, ptr %D, align 4
  ret void
}
;; CHECK: mul_and_sub_4
;; CHECK: vsub(v{{[0-9]+}}.qf16,v{{[0-9]+}}.hf)
