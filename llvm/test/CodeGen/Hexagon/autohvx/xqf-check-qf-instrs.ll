; For non-legacy mode, the presence of qf instructions is not checked. Post RA pass is run and
; V30 register is reserved.
; For legacy mode, it is checked if the function contains qf generating instructions, only
; then V30 register is reserved and the postRA pass run.

; REQUIRES: asserts
; RUN: llc -mtriple=hexagon-unknown-elf -mhvx -mcpu=hexagonv79 -mattr=+hvxv79,+hvx-length128b,+hvx-qfloat -enable-xqf-gen=true \
; RUN: -hexagon-qfloat-mode=ieee -debug-only=handle-qfp -o /dev/null < %s \
; RUN: 2>&1 | FileCheck %s --check-prefix=IEEE
; RUN: llc -mtriple=hexagon-unknown-elf -mhvx -mcpu=hexagonv81 -mattr=+hvxv81,+hvx-length128b,+hvx-qfloat -enable-xqf-gen=true \
; RUN: -hexagon-qfloat-mode=ieee -debug-only=handle-qfp -o /dev/null < %s \
; RUN: 2>&1 | FileCheck %s --check-prefix=IEEE
; RUN: llc -mtriple=hexagon-unknown-elf -mhvx -mcpu=hexagonv79 -mattr=+hvxv79,+hvx-length128b,+hvx-qfloat -enable-xqf-gen=true \
; RUN: -hexagon-qfloat-mode=strict-ieee -debug-only=handle-qfp -o /dev/null < %s \
; RUN: 2>&1 | FileCheck %s --check-prefix=STRICT
; RUN: llc -mtriple=hexagon-unknown-elf -mhvx -mcpu=hexagonv81 -mattr=+hvxv81,+hvx-length128b,+hvx-qfloat -enable-xqf-gen=true \
; RUN: -hexagon-qfloat-mode=strict-ieee -debug-only=handle-qfp -o /dev/null < %s \
; RUN: 2>&1 | FileCheck %s --check-prefix=STRICT
; RUN: llc -mtriple=hexagon-unknown-elf -mhvx -mcpu=hexagonv79 -mattr=+hvxv79,+hvx-length128b,+hvx-qfloat -enable-xqf-gen=true \
; RUN: -hexagon-qfloat-mode=lossy -debug-only=handle-qfp -o /dev/null < %s \
; RUN: 2>&1 | FileCheck %s --check-prefix=LOSSY
; RUN: llc -mtriple=hexagon-unknown-elf -mhvx -mcpu=hexagonv81 -mattr=+hvxv81,+hvx-length128b,+hvx-qfloat -enable-xqf-gen=true \
; RUN: -hexagon-qfloat-mode=lossy -debug-only=handle-qfp -o /dev/null < %s \
; RUN: 2>&1 | FileCheck %s --check-prefix=LOSSY

; RUN: llc -mtriple=hexagon-unknown-elf -mhvx -mcpu=hexagonv79 -mattr=+hvxv79,+hvx-length128b,+hvx-qfloat -enable-xqf-gen=true \
; RUN: -debug-only=handle-qfp -o /dev/null < %s \
; RUN: 2>&1 | FileCheck %s --check-prefix=LEGACY
; RUN: llc -mtriple=hexagon-unknown-elf -mhvx -mcpu=hexagonv81 -mattr=+hvxv81,+hvx-length128b,+hvx-qfloat -enable-xqf-gen=true \
; RUN: -debug-only=handle-qfp -o /dev/null < %s \
; RUN: 2>&1 | FileCheck %s --check-prefix=LEGACY

define dso_local <32 x i32> @test1(i32 noundef %input1, i32 noundef %input2, i32 noundef %size) local_unnamed_addr #0 {
; IEEE: Entering Hexagon Fixup QF spills and refills pass
; STRICT: Entering Hexagon Fixup QF spills and refills pass
; LOSSY: Entering Hexagon Fixup QF spills and refills pass
; LEGACY: Entering Hexagon Fixup QF spills and refills pass
; IEEE : Handling spills
; STRICT: Handling spills
; LOSSY: Handling spills
; LEGACY: Handling spills
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32 %input1)
  %1 = tail call <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32 %input2)
  %2 = tail call <32 x i32> @llvm.hexagon.V6.vadd.sf.128B(<32 x i32> %0, <32 x i32> %1)
  %3 = tail call <32 x i32> @llvm.hexagon.V6.vconv.sf.qf32.128B(<32 x i32> %2)
  %4 = tail call <32 x i32> @llvm.hexagon.V6.vsub.sf.128B(<32 x i32> %0, <32 x i32> %3)
  ret <32 x i32> %4
}

define dso_local <32 x i32> @test2(i32 noundef %input1, i32 noundef %input2, i32 noundef %size) local_unnamed_addr #0 {
; IEEE: Entering Hexagon Fixup QF spills and refills pass
; STRICT: Entering Hexagon Fixup QF spills and refills pass
; LOSSY: Entering Hexagon Fixup QF spills and refills pass
; LEGACY: Entering Hexagon Fixup QF spills and refills pass
; IEEE: Handling spills
; STRICT: Handling spills
; LOSSY: Handling spills
; LEGACY-NOT: Handling spills
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32 %input1)
  %1 = tail call <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32 %input2)
  %2 = add nsw <32 x i32> %0, %1
  ret <32 x i32> %2
}

declare <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32) #1
declare <32 x i32> @llvm.hexagon.V6.vadd.sf.128B(<32 x i32>, <32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vconv.sf.qf32.128B(<32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vsub.sf.128B(<32 x i32>, <32 x i32>) #1

attributes #0 = { nounwind }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(none) }
