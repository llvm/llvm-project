; RUN: llc -march=hexagon < %s | FileCheck %s

; This testcase is known to generate an opportunity for creating vcombine
; in HexagonCopyToCombine.

; CHECK: vcombine

target triple = "hexagon-unknown--elf"

declare <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vabsdiffuh.128B(<32 x i32>, <32 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vlalignbi.128B(<32 x i32>, <32 x i32>, i32) #0
declare <32 x i32> @llvm.hexagon.V6.vsathub.128B(<32 x i32>, <32 x i32>) #0
declare <64 x i32> @llvm.hexagon.V6.vaddh.dv.128B(<64 x i32>, <64 x i32>) #0
declare <64 x i32> @llvm.hexagon.V6.vadduhsat.dv.128B(<64 x i32>, <64 x i32>) #0
declare <64 x i32> @llvm.hexagon.V6.vaddubh.128B(<32 x i32>, <32 x i32>) #0
declare <64 x i32> @llvm.hexagon.V6.vmpyub.128B(<32 x i32>, i32) #0

define void @foo() local_unnamed_addr #1 {
entry:
  %0 = load <32 x i32>, <32 x i32>* undef, align 128
  %1 = load <32 x i32>, <32 x i32>* null, align 128
  br i1 undef, label %b2, label %b1

b1:                                                                ; preds = %entry
  %2 = tail call <32 x i32> @llvm.hexagon.V6.vlalignbi.128B(<32 x i32> %0, <32 x i32> %1, i32 1)
  %3 = tail call <64 x i32> @llvm.hexagon.V6.vmpyub.128B(<32 x i32> %2, i32 33686018) #1
  %4 = tail call <64 x i32> @llvm.hexagon.V6.vadduhsat.dv.128B(<64 x i32> undef, <64 x i32> %3) #1
  %5 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %4)
  %6 = tail call <32 x i32> @llvm.hexagon.V6.vabsdiffuh.128B(<32 x i32> %5, <32 x i32> undef) #1
  %7 = tail call <64 x i32> @llvm.hexagon.V6.vaddubh.128B(<32 x i32> %6, <32 x i32> undef)
  %8 = tail call <64 x i32> @llvm.hexagon.V6.vaddh.dv.128B(<64 x i32> undef, <64 x i32> %7) #1
  %9 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %8) #1
  %10 = tail call <32 x i32> @llvm.hexagon.V6.vsathub.128B(<32 x i32> %9, <32 x i32> undef) #1
  store <32 x i32> %10, <32 x i32>* undef, align 128
  br label %b2

b2:                                                                ; preds = %b1, %entry
  %c2.host31.sroa.3.2.unr.ph = phi <32 x i32> [ zeroinitializer, %b1 ], [ %0, %entry ]
  %c2.host31.sroa.0.2.unr.ph = phi <32 x i32> [ %0, %b1 ], [ %1, %entry ]
  %11 = tail call <32 x i32> @llvm.hexagon.V6.vlalignbi.128B(<32 x i32> %c2.host31.sroa.3.2.unr.ph, <32 x i32> %c2.host31.sroa.0.2.unr.ph, i32 1)
  %12 = tail call <64 x i32> @llvm.hexagon.V6.vmpyub.128B(<32 x i32> %11, i32 33686018) #1
  %13 = tail call <64 x i32> @llvm.hexagon.V6.vadduhsat.dv.128B(<64 x i32> undef, <64 x i32> %12) #1
  %14 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %13)
  %15 = tail call <32 x i32> @llvm.hexagon.V6.vabsdiffuh.128B(<32 x i32> %14, <32 x i32> undef) #1
  %16 = tail call <64 x i32> @llvm.hexagon.V6.vaddubh.128B(<32 x i32> %15, <32 x i32> undef)
  %17 = tail call <64 x i32> @llvm.hexagon.V6.vaddh.dv.128B(<64 x i32> undef, <64 x i32> %16) #1
  %18 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %17) #1
  %19 = tail call <32 x i32> @llvm.hexagon.V6.vsathub.128B(<32 x i32> %18, <32 x i32> undef) #1
  store <32 x i32> %19, <32 x i32>* undef, align 128
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-double" }

