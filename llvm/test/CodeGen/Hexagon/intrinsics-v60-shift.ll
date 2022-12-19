; RUN: llc -march=hexagon < %s | FileCheck %s

@d = external global <16 x i32>

; CHECK-LABEL: test18:
; CHECK: v{{[0-9]+}}.uw = vcl0(v{{[0-9]+}}.uw)
define void @test18(<16 x i32> %a) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vcl0w(<16 x i32> %a)
  store <16 x i32> %0, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test19:
; CHECK: v{{[0-9]+}}.h = vpopcount(v{{[0-9]+}}.h)
define void @test19(<16 x i32> %a) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vpopcounth(<16 x i32> %a)
  store <16 x i32> %0, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test20:
; CHECK: v{{[0-9]+}}.uh = vcl0(v{{[0-9]+}}.uh)
define void @test20(<16 x i32> %a) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vcl0h(<16 x i32> %a)
  store <16 x i32> %0, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test21:
; CHECK: v{{[0-9]+}}.w = vnormamt(v{{[0-9]+}}.w)
define void @test21(<16 x i32> %a) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vnormamtw(<16 x i32> %a)
  store <16 x i32> %0, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test22:
; CHECK: v{{[0-9]+}}.h = vnormamt(v{{[0-9]+}}.h)
define void @test22(<16 x i32> %a) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vnormamth(<16 x i32> %a)
  store <16 x i32> %0, ptr @d, align 64
  ret void
}

declare <16 x i32> @llvm.hexagon.V6.vcl0w(<16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vpopcounth(<16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vcl0h(<16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vnormamtw(<16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vnormamth(<16 x i32>) #0

attributes #0 = { nounwind readnone "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
