; RUN: llc < %s -mattr=+avx -mtriple=i686-pc-win32 | FileCheck %s

define void @endless_loop() {
; CHECK-LABEL: endless_loop:
; CHECK-NEXT:  # BB#0:
; CHECK-NEXT:    vbroadcastss (%eax), %ymm0
; CHECK-NEXT:    vmovddup {{.*#+}} xmm1 = xmm0[0,0]
; CHECK-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm1
; CHECK-NEXT:    vxorps %xmm2, %xmm2, %xmm2
; CHECK-NEXT:    vblendps {{.*#+}} ymm1 = ymm2[0,1,2,3,4,5,6],ymm1[7]
; CHECK-NEXT:    vxorps %ymm2, %ymm2, %ymm2
; CHECK-NEXT:    vblendps {{.*#+}} ymm0 = ymm0[0],ymm2[1,2,3,4,5,6,7]
; CHECK-NEXT:    vmovaps %ymm0, (%eax)
; CHECK-NEXT:    vmovaps %ymm1, (%eax)
; CHECK-NEXT:    vzeroupper
; CHECK-NEXT:    retl
entry:
  %0 = load <8 x i32>, <8 x i32> addrspace(1)* undef, align 32
  %1 = shufflevector <8 x i32> %0, <8 x i32> undef, <16 x i32> <i32 4, i32 4, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %2 = shufflevector <16 x i32> <i32 undef, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 undef>, <16 x i32> %1, <16 x i32> <i32 16, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 17>
  store <16 x i32> %2, <16 x i32> addrspace(1)* undef, align 64
  ret void
}
