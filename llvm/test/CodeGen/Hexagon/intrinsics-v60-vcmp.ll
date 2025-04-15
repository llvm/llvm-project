; RUN: llc -mtriple=hexagon < %s | FileCheck %s

@d = external global <16 x i32>, align 64

; CHECK-LABEL: test1:
; CHECK: q{{[0-9]}} &= vcmp.eq(v{{[0-9]+}}.b,v{{[0-9]+}}.b)
define void @test1(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %v0 = load <16 x i32>, ptr @d, align 64
  %v1 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %v0, i32 -1)
  %v2 = tail call <64 x i1> @llvm.hexagon.V6.veqb.and(<64 x i1> %v1, <16 x i32> %a, <16 x i32> %b)
  %v3 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1> %v2, i32 -1)
  store <16 x i32> %v3, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test2:
; CHECK: q{{[0-9]}} &= vcmp.eq(v{{[0-9]+}}.h,v{{[0-9]+}}.h)
define void @test2(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %v0 = load <16 x i32>, ptr @d, align 64
  %v1 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %v0, i32 -1)
  %v2 = tail call <64 x i1> @llvm.hexagon.V6.veqh.and(<64 x i1> %v1, <16 x i32> %a, <16 x i32> %b)
  %v3 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1> %v2, i32 -1)
  store <16 x i32> %v3, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test3:
; CHECK: q{{[0-9]}} &= vcmp.eq(v{{[0-9]+}}.w,v{{[0-9]+}}.w)
define void @test3(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %v0 = load <16 x i32>, ptr @d, align 64
  %v1 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %v0, i32 -1)
  %v2 = tail call <64 x i1> @llvm.hexagon.V6.veqw.and(<64 x i1> %v1, <16 x i32> %a, <16 x i32> %b)
  %v3 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1> %v2, i32 -1)
  store <16 x i32> %v3, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test4:
; CHECK: q{{[0-9]}} &= vcmp.gt(v{{[0-9]+}}.b,v{{[0-9]+}}.b)
define void @test4(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %v0 = load <16 x i32>, ptr @d, align 64
  %v1 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %v0, i32 -1)
  %v2 = tail call <64 x i1> @llvm.hexagon.V6.vgtb.and(<64 x i1> %v1, <16 x i32> %a, <16 x i32> %b)
  %v3 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1> %v2, i32 -1)
  store <16 x i32> %v3, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test5:
; CHECK: q{{[0-9]}} &= vcmp.gt(v{{[0-9]+}}.h,v{{[0-9]+}}.h)
define void @test5(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %v0 = load <16 x i32>, ptr @d, align 64
  %v1 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %v0, i32 -1)
  %v2 = tail call <64 x i1> @llvm.hexagon.V6.vgth.and(<64 x i1> %v1, <16 x i32> %a, <16 x i32> %b)
  %v3 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1> %v2, i32 -1)
  store <16 x i32> %v3, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test6:
; CHECK: q{{[0-9]}} &= vcmp.gt(v{{[0-9]+}}.w,v{{[0-9]+}}.w)
define void @test6(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %v0 = load <16 x i32>, ptr @d, align 64
  %v1 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %v0, i32 -1)
  %v2 = tail call <64 x i1> @llvm.hexagon.V6.vgtw.and(<64 x i1> %v1, <16 x i32> %a, <16 x i32> %b)
  %v3 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1> %v2, i32 -1)
  store <16 x i32> %v3, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test7:
; CHECK: q{{[0-9]}} &= vcmp.gt(v{{[0-9]+}}.ub,v{{[0-9]+}}.ub)
define void @test7(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %v0 = load <16 x i32>, ptr @d, align 64
  %v1 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %v0, i32 -1)
  %v2 = tail call <64 x i1> @llvm.hexagon.V6.vgtub.and(<64 x i1> %v1, <16 x i32> %a, <16 x i32> %b)
  %v3 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1> %v2, i32 -1)
  store <16 x i32> %v3, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test8:
; CHECK: q{{[0-9]}} &= vcmp.gt(v{{[0-9]+}}.uh,v{{[0-9]+}}.uh)
define void @test8(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %v0 = load <16 x i32>, ptr @d, align 64
  %v1 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %v0, i32 -1)
  %v2 = tail call <64 x i1> @llvm.hexagon.V6.vgtuh.and(<64 x i1> %v1, <16 x i32> %a, <16 x i32> %b)
  %v3 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1> %v2, i32 -1)
  store <16 x i32> %v3, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test9:
; CHECK: q{{[0-9]}} &= vcmp.gt(v{{[0-9]+}}.uw,v{{[0-9]+}}.uw)
define void @test9(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %v0 = load <16 x i32>, ptr @d, align 64
  %v1 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %v0, i32 -1)
  %v2 = tail call <64 x i1> @llvm.hexagon.V6.vgtuw.and(<64 x i1> %v1, <16 x i32> %a, <16 x i32> %b)
  %v3 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1> %v2, i32 -1)
  store <16 x i32> %v3, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test10:
; CHECK: q{{[0-9]}} |= vcmp.eq(v{{[0-9]+}}.b,v{{[0-9]+}}.b)
define void @test10(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %v0 = load <16 x i32>, ptr @d, align 64
  %v1 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %v0, i32 -1)
  %v2 = tail call <64 x i1> @llvm.hexagon.V6.veqb.or(<64 x i1> %v1, <16 x i32> %a, <16 x i32> %b)
  %v3 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1> %v2, i32 -1)
  store <16 x i32> %v3, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test11:
; CHECK: q{{[0-9]}} |= vcmp.eq(v{{[0-9]+}}.h,v{{[0-9]+}}.h)
define void @test11(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %v0 = load <16 x i32>, ptr @d, align 64
  %v1 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %v0, i32 -1)
  %v2 = tail call <64 x i1> @llvm.hexagon.V6.veqh.or(<64 x i1> %v1, <16 x i32> %a, <16 x i32> %b)
  %v3 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1> %v2, i32 -1)
  store <16 x i32> %v3, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test12:
; CHECK: q{{[0-9]}} |= vcmp.eq(v{{[0-9]+}}.w,v{{[0-9]+}}.w)
define void @test12(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %v0 = load <16 x i32>, ptr @d, align 64
  %v1 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %v0, i32 -1)
  %v2 = tail call <64 x i1> @llvm.hexagon.V6.veqw.or(<64 x i1> %v1, <16 x i32> %a, <16 x i32> %b)
  %v3 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1> %v2, i32 -1)
  store <16 x i32> %v3, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test13:
; CHECK: q{{[0-9]}} |= vcmp.gt(v{{[0-9]+}}.b,v{{[0-9]+}}.b)
define void @test13(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %v0 = load <16 x i32>, ptr @d, align 64
  %v1 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %v0, i32 -1)
  %v2 = tail call <64 x i1> @llvm.hexagon.V6.vgtb.or(<64 x i1> %v1, <16 x i32> %a, <16 x i32> %b)
  %v3 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1> %v2, i32 -1)
  store <16 x i32> %v3, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test14:
; CHECK: q{{[0-9]}} |= vcmp.gt(v{{[0-9]+}}.h,v{{[0-9]+}}.h)
define void @test14(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %v0 = load <16 x i32>, ptr @d, align 64
  %v1 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %v0, i32 -1)
  %v2 = tail call <64 x i1> @llvm.hexagon.V6.vgth.or(<64 x i1> %v1, <16 x i32> %a, <16 x i32> %b)
  %v3 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1> %v2, i32 -1)
  store <16 x i32> %v3, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test15:
; CHECK: q{{[0-9]}} |= vcmp.gt(v{{[0-9]+}}.w,v{{[0-9]+}}.w)
define void @test15(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %v0 = load <16 x i32>, ptr @d, align 64
  %v1 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %v0, i32 -1)
  %v2 = tail call <64 x i1> @llvm.hexagon.V6.vgtw.or(<64 x i1> %v1, <16 x i32> %a, <16 x i32> %b)
  %v3 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1> %v2, i32 -1)
  store <16 x i32> %v3, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test16:
; CHECK: q{{[0-9]}} |= vcmp.gt(v{{[0-9]+}}.ub,v{{[0-9]+}}.ub)
define void @test16(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %v0 = load <16 x i32>, ptr @d, align 64
  %v1 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %v0, i32 -1)
  %v2 = tail call <64 x i1> @llvm.hexagon.V6.vgtub.or(<64 x i1> %v1, <16 x i32> %a, <16 x i32> %b)
  %v3 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1> %v2, i32 -1)
  store <16 x i32> %v3, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test17:
; CHECK: q{{[0-9]}} |= vcmp.gt(v{{[0-9]+}}.uh,v{{[0-9]+}}.uh)
define void @test17(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %v0 = load <16 x i32>, ptr @d, align 64
  %v1 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %v0, i32 -1)
  %v2 = tail call <64 x i1> @llvm.hexagon.V6.vgtuh.or(<64 x i1> %v1, <16 x i32> %a, <16 x i32> %b)
  %v3 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1> %v2, i32 -1)
  store <16 x i32> %v3, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test18:
; CHECK: q{{[0-9]}} |= vcmp.gt(v{{[0-9]+}}.uw,v{{[0-9]+}}.uw)
define void @test18(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %v0 = load <16 x i32>, ptr @d, align 64
  %v1 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %v0, i32 -1)
  %v2 = tail call <64 x i1> @llvm.hexagon.V6.vgtuw.or(<64 x i1> %v1, <16 x i32> %a, <16 x i32> %b)
  %v3 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1> %v2, i32 -1)
  store <16 x i32> %v3, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test19:
; CHECK: q{{[0-9]}} ^= vcmp.eq(v{{[0-9]+}}.b,v{{[0-9]+}}.b)
define void @test19(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %v0 = load <16 x i32>, ptr @d, align 64
  %v1 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %v0, i32 -1)
  %v2 = tail call <64 x i1> @llvm.hexagon.V6.veqb.xor(<64 x i1> %v1, <16 x i32> %a, <16 x i32> %b)
  %v3 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1> %v2, i32 -1)
  store <16 x i32> %v3, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test20:
; CHECK: q{{[0-9]}} ^= vcmp.eq(v{{[0-9]+}}.h,v{{[0-9]+}}.h)
define void @test20(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %v0 = load <16 x i32>, ptr @d, align 64
  %v1 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %v0, i32 -1)
  %v2 = tail call <64 x i1> @llvm.hexagon.V6.veqh.xor(<64 x i1> %v1, <16 x i32> %a, <16 x i32> %b)
  %v3 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1> %v2, i32 -1)
  store <16 x i32> %v3, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test21:
; CHECK: q{{[0-9]}} ^= vcmp.eq(v{{[0-9]+}}.w,v{{[0-9]+}}.w)
define void @test21(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %v0 = load <16 x i32>, ptr @d, align 64
  %v1 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %v0, i32 -1)
  %v2 = tail call <64 x i1> @llvm.hexagon.V6.veqw.xor(<64 x i1> %v1, <16 x i32> %a, <16 x i32> %b)
  %v3 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1> %v2, i32 -1)
  store <16 x i32> %v3, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test22:
; CHECK: q{{[0-9]}} ^= vcmp.gt(v{{[0-9]+}}.b,v{{[0-9]+}}.b)
define void @test22(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %v0 = load <16 x i32>, ptr @d, align 64
  %v1 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %v0, i32 -1)
  %v2 = tail call <64 x i1> @llvm.hexagon.V6.vgtb.xor(<64 x i1> %v1, <16 x i32> %a, <16 x i32> %b)
  %v3 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1> %v2, i32 -1)
  store <16 x i32> %v3, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test23:
; CHECK: q{{[0-9]}} ^= vcmp.gt(v{{[0-9]+}}.h,v{{[0-9]+}}.h)
define void @test23(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %v0 = load <16 x i32>, ptr @d, align 64
  %v1 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %v0, i32 -1)
  %v2 = tail call <64 x i1> @llvm.hexagon.V6.vgth.xor(<64 x i1> %v1, <16 x i32> %a, <16 x i32> %b)
  %v3 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1> %v2, i32 -1)
  store <16 x i32> %v3, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test24:
; CHECK: q{{[0-9]}} ^= vcmp.gt(v{{[0-9]+}}.w,v{{[0-9]+}}.w)
define void @test24(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %v0 = load <16 x i32>, ptr @d, align 64
  %v1 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %v0, i32 -1)
  %v2 = tail call <64 x i1> @llvm.hexagon.V6.vgtw.xor(<64 x i1> %v1, <16 x i32> %a, <16 x i32> %b)
  %v3 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1> %v2, i32 -1)
  store <16 x i32> %v3, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test25:
; CHECK: q{{[0-9]}} ^= vcmp.gt(v{{[0-9]+}}.ub,v{{[0-9]+}}.ub)
define void @test25(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %v0 = load <16 x i32>, ptr @d, align 64
  %v1 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %v0, i32 -1)
  %v2 = tail call <64 x i1> @llvm.hexagon.V6.vgtub.xor(<64 x i1> %v1, <16 x i32> %a, <16 x i32> %b)
  %v3 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1> %v2, i32 -1)
  store <16 x i32> %v3, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test26:
; CHECK: q{{[0-9]}} ^= vcmp.gt(v{{[0-9]+}}.uh,v{{[0-9]+}}.uh)
define void @test26(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %v0 = load <16 x i32>, ptr @d, align 64
  %v1 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %v0, i32 -1)
  %v2 = tail call <64 x i1> @llvm.hexagon.V6.vgtuh.xor(<64 x i1> %v1, <16 x i32> %a, <16 x i32> %b)
  %v3 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1> %v2, i32 -1)
  store <16 x i32> %v3, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test27:
; CHECK: q{{[0-9]}} ^= vcmp.gt(v{{[0-9]+}}.uw,v{{[0-9]+}}.uw)
define void @test27(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %v0 = load <16 x i32>, ptr @d, align 64
  %v1 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %v0, i32 -1)
  %v2 = tail call <64 x i1> @llvm.hexagon.V6.vgtuw.xor(<64 x i1> %v1, <16 x i32> %a, <16 x i32> %b)
  %v3 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1> %v2, i32 -1)
  store <16 x i32> %v3, ptr @d, align 64
  ret void
}

declare <64 x i1> @llvm.hexagon.V6.veqb.and(<64 x i1>, <16 x i32>, <16 x i32>) #0
declare <64 x i1> @llvm.hexagon.V6.veqh.and(<64 x i1>, <16 x i32>, <16 x i32>) #0
declare <64 x i1> @llvm.hexagon.V6.veqw.and(<64 x i1>, <16 x i32>, <16 x i32>) #0
declare <64 x i1> @llvm.hexagon.V6.vgtb.and(<64 x i1>, <16 x i32>, <16 x i32>) #0
declare <64 x i1> @llvm.hexagon.V6.vgth.and(<64 x i1>, <16 x i32>, <16 x i32>) #0
declare <64 x i1> @llvm.hexagon.V6.vgtw.and(<64 x i1>, <16 x i32>, <16 x i32>) #0
declare <64 x i1> @llvm.hexagon.V6.vgtub.and(<64 x i1>, <16 x i32>, <16 x i32>) #0
declare <64 x i1> @llvm.hexagon.V6.vgtuh.and(<64 x i1>, <16 x i32>, <16 x i32>) #0
declare <64 x i1> @llvm.hexagon.V6.vgtuw.and(<64 x i1>, <16 x i32>, <16 x i32>) #0
declare <64 x i1> @llvm.hexagon.V6.veqb.or(<64 x i1>, <16 x i32>, <16 x i32>) #0
declare <64 x i1> @llvm.hexagon.V6.veqh.or(<64 x i1>, <16 x i32>, <16 x i32>) #0
declare <64 x i1> @llvm.hexagon.V6.veqw.or(<64 x i1>, <16 x i32>, <16 x i32>) #0
declare <64 x i1> @llvm.hexagon.V6.vgtb.or(<64 x i1>, <16 x i32>, <16 x i32>) #0
declare <64 x i1> @llvm.hexagon.V6.vgth.or(<64 x i1>, <16 x i32>, <16 x i32>) #0
declare <64 x i1> @llvm.hexagon.V6.vgtw.or(<64 x i1>, <16 x i32>, <16 x i32>) #0
declare <64 x i1> @llvm.hexagon.V6.vgtub.or(<64 x i1>, <16 x i32>, <16 x i32>) #0
declare <64 x i1> @llvm.hexagon.V6.vgtuh.or(<64 x i1>, <16 x i32>, <16 x i32>) #0
declare <64 x i1> @llvm.hexagon.V6.vgtuw.or(<64 x i1>, <16 x i32>, <16 x i32>) #0
declare <64 x i1> @llvm.hexagon.V6.veqb.xor(<64 x i1>, <16 x i32>, <16 x i32>) #0
declare <64 x i1> @llvm.hexagon.V6.veqh.xor(<64 x i1>, <16 x i32>, <16 x i32>) #0
declare <64 x i1> @llvm.hexagon.V6.veqw.xor(<64 x i1>, <16 x i32>, <16 x i32>) #0
declare <64 x i1> @llvm.hexagon.V6.vgtb.xor(<64 x i1>, <16 x i32>, <16 x i32>) #0
declare <64 x i1> @llvm.hexagon.V6.vgth.xor(<64 x i1>, <16 x i32>, <16 x i32>) #0
declare <64 x i1> @llvm.hexagon.V6.vgtw.xor(<64 x i1>, <16 x i32>, <16 x i32>) #0
declare <64 x i1> @llvm.hexagon.V6.vgtub.xor(<64 x i1>, <16 x i32>, <16 x i32>) #0
declare <64 x i1> @llvm.hexagon.V6.vgtuh.xor(<64 x i1>, <16 x i32>, <16 x i32>) #0
declare <64 x i1> @llvm.hexagon.V6.vgtuw.xor(<64 x i1>, <16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1>, i32) #0
declare <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32>, i32) #0

attributes #0 = { nounwind readnone "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
