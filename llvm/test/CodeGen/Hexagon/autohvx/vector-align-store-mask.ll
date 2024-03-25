; RUN: llc -march=hexagon -hexagon-hvx-widen=32 < %s | FileCheck %s

target triple = "hexagon"

; Test the the store mask is adjusted for gaps between sections. The
; Vector Combine pass generates masked stores for chunks of 128 bytes.
; The masked store must be shifted if the first store in a section
; is not a multiple of 128 bytes. This test checks that two masks
; are created, and the second mask is used in a masked store.

; CHECK: [[REG:r[0-9]+]] = ##.LCPI0_1
; CHECK: [[VREG1:v[0-9]+]] = vmem([[REG]]+#0)
; CHECK: [[VREG2:v[0-9]+]] = vlalign([[VREG1]],v{{[0-9]+}},r{{[0-9]+}})
; CHECK: [[QREG:q[0-3]+]] = vand([[VREG2]],r{{[0-9]+}})
; CHECK: if ([[QREG]]) vmem({{.*}}) = v{{[0-9]+}}

define dllexport void @f0(ptr %a0) local_unnamed_addr #0 {
b0:
  br label %b1

b1:                                               ; preds = %b1, %b0
  %v0 = or i32 -1, 40
  %v1 = getelementptr inbounds i32, ptr %a0, i32 %v0
  store <8 x i32> undef, ptr %v1, align 32
  %v3 = or i32 0, 48
  %v4 = getelementptr inbounds i32, ptr %a0, i32 %v3
  store <8 x i32> undef, ptr %v4, align 64
  br i1 undef, label %b2, label %b1

b2:                                               ; preds = %b1
  ret void
}

attributes #0 = { "target-features"="+hvxv66,+hvx-length128b,-packets" }
