; RUN: llc -march=hexagon < %s | FileCheck %s

@d = external global <16 x i32>
@c = external global <32 x i32>

; CHECK-LABEL: test1:
; CHECK: v{{[0-9]+}}.b = vpacke(v{{[0-9]+}}.h,v{{[0-9]+}}.h)
define void @test1(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vpackeb(<16 x i32> %a, <16 x i32> %b)
  store <16 x i32> %0, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test2:
; CHECK: v{{[0-9]+}}.h = vpacke(v{{[0-9]+}}.w,v{{[0-9]+}}.w)
define void @test2(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vpackeh(<16 x i32> %a, <16 x i32> %b)
  store <16 x i32> %0, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test3:
; CHECK: v{{[0-9]+}}.ub = vpack(v{{[0-9]+}}.h,v{{[0-9]+}}.h):sat
define void @test3(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vpackhub.sat(<16 x i32> %a, <16 x i32> %b)
  store <16 x i32> %0, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test4:
; CHECK: v{{[0-9]+}}.b = vpack(v{{[0-9]+}}.h,v{{[0-9]+}}.h):sat
define void @test4(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vpackhb.sat(<16 x i32> %a, <16 x i32> %b)
  store <16 x i32> %0, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test5:
; CHECK: v{{[0-9]+}}.uh = vpack(v{{[0-9]+}}.w,v{{[0-9]+}}.w):sat
define void @test5(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vpackwuh.sat(<16 x i32> %a, <16 x i32> %b)
  store <16 x i32> %0, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test6:
; CHECK: v{{[0-9]+}}.h = vpack(v{{[0-9]+}}.w,v{{[0-9]+}}.w):sat
define void @test6(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vpackwh.sat(<16 x i32> %a, <16 x i32> %b)
  store <16 x i32> %0, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test7:
; CHECK: v{{[0-9]+}}.b = vpacko(v{{[0-9]+}}.h,v{{[0-9]+}}.h)
define void @test7(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vpackob(<16 x i32> %a, <16 x i32> %b)
  store <16 x i32> %0, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test8:
; CHECK: v{{[0-9]+}}.h = vpacko(v{{[0-9]+}}.w,v{{[0-9]+}}.w)
define void @test8(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vpackoh(<16 x i32> %a, <16 x i32> %b)
  store <16 x i32> %0, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test9:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.uh = vunpack(v{{[0-9]+}}.ub)
define void @test9(<16 x i32> %a) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vunpackub(<16 x i32> %a)
  store <32 x i32> %0, ptr @c, align 128
  ret void
}

; CHECK-LABEL: test10:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.uw = vunpack(v{{[0-9]+}}.uh)
define void @test10(<16 x i32> %a) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vunpackuh(<16 x i32> %a)
  store <32 x i32> %0, ptr @c, align 128
  ret void
}

; CHECK-LABEL: test11:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.h = vunpack(v{{[0-9]+}}.b)
define void @test11(<16 x i32> %a) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vunpackb(<16 x i32> %a)
  store <32 x i32> %0, ptr @c, align 128
  ret void
}

; CHECK-LABEL: test12:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.w = vunpack(v{{[0-9]+}}.h)
define void @test12(<16 x i32> %a) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vunpackh(<16 x i32> %a)
  store <32 x i32> %0, ptr @c, align 128
  ret void
}

; CHECK-LABEL: test13:
; CHECK: v{{[0-9]+}}.h = vdeal(v{{[0-9]+}}.h)
define void @test13(<16 x i32> %a) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vdealh(<16 x i32> %a)
  store <16 x i32> %0, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test14:
; CHECK: v{{[0-9]+}}.b = vdeal(v{{[0-9]+}}.b)
define void @test14(<16 x i32> %a) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vdealb(<16 x i32> %a)
  store <16 x i32> %0, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test15:
; CHECK: v{{[0-9]+}}.h = vshuff(v{{[0-9]+}}.h)
define void @test15(<16 x i32> %a) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vshuffh(<16 x i32> %a)
  store <16 x i32> %0, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test16:
; CHECK: v{{[0-9]+}}.b = vshuff(v{{[0-9]+}}.b)
define void @test16(<16 x i32> %a) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vshuffb(<16 x i32> %a)
  store <16 x i32> %0, ptr @d, align 64
  ret void
}

declare <16 x i32> @llvm.hexagon.V6.vpackeb(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vpackeh(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vpackhub.sat(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vpackhb.sat(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vpackwuh.sat(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vpackwh.sat(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vpackob(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vpackoh(<16 x i32>, <16 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vunpackub(<16 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vunpackuh(<16 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vunpackb(<16 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vunpackh(<16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vdealh(<16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vdealb(<16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vshuffh(<16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vshuffb(<16 x i32>) #0

attributes #0 = { nounwind readnone "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
