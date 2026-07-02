; Tests for emitted warnings when IEEE type is used as qf and vice-versa
; post register allocation.

; REQUIRES: asserts
; RUN: llc --mtriple=hexagon-- -mhvx -mcpu=hexagonv79 -mattr=+hvxv79,+hvx-length128b,+hvx-qfloat \
; RUN: -enable-xqf-gen=false -enable-postra-xqf-check=true -hexagon-qfloat-mode=ieee \
; RUN: -verify-machineinstrs 2>&1 < %s -o /dev/null | FileCheck %s
; RUN: llc --mtriple=hexagon-- -mhvx -mcpu=hexagonv81 -mattr=+hvxv81,+hvx-length128b,+hvx-qfloat \
; RUN: -enable-xqf-gen=false -enable-postra-xqf-check=true -hexagon-qfloat-mode=ieee \
; RUN: -verify-machineinstrs 2>&1 < %s -o /dev/null | FileCheck %s

define dso_local inreg <64 x i32> @foo(<32 x i32> noundef %vina, <32 x i32> noundef %vinb) local_unnamed_addr #0{
;CHECK: Mismatch: hf type used as qf16 at operand 1
;CHECK-NEXT: Def: renamable [[VREG2:\$v[0-9]+]] = V6_lvsplath
;CHECK-NEXT: Use: renamable $v{{[0-9]+}} = V6_vadd_qf16_mix killed renamable [[VREG2]]
;CHECK-NEXT: Mismatch: sf type used as qf32 at operand 1
;CHECK-NEXT: Def: renamable [[VREG4:\$v[0-9]+]] = V6_lvsplatw
;CHECK-NEXT: Use: renamable $v{{[0-9]+}} = V6_vadd_qf32_mix killed renamable [[VREG4]]
;CHECK-NEXT: Mismatch: qf16 type used as hf at operand 2
;CHECK-NEXT: Def: renamable [[VREG6:\$v[0-9]+]] = V6_vadd_qf16_mix
;CHECK-NEXT: Use: renamable $w{{[0-9]+}} = V6_vmpy_qf32_hf killed renamable $v{{[0-9]+}}, killed renamable [[VREG6]]
;CHECK-NEXT: Mismatch: qf32 type used as sf at operand 2
;CHECK-NEXT: Def: renamable [[VREG7:\$v[0-9]+]] = V6_vadd_qf32_mix
;CHECK-NEXT: Use: renamable $v{{[0-9]+}} = V6_vsub_qf32_mix killed renamable $v{{[0-9]+}}, renamable [[VREG7]]
;CHECK: Mismatch: qf32 type used as sf at operand 2
;CHECK: Def: renamable $v[[R0:[0-9]+]] = V6_vsub_qf32_mix
;CHECK: Mismatch: qf32 type used as sf at operand 2
;CHECK: Def: renamable $v[[R1:[0-9]+]] = V6_vsub_qf32
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.lvsplath.128B(i32 15360)
  %1 = tail call <32 x i32> @llvm.hexagon.V6.lvsplath.128B(i32 48128)
  %2 = tail call <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32 44032)
  %3 = tail call <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32 56320)
  %4 = tail call <32 x i32> @llvm.hexagon.V6.vadd.qf16.mix.128B(<32 x i32> %0, <32 x i32> %1)
  %5 = tail call <32 x i32> @llvm.hexagon.V6.vadd.qf32.mix.128B(<32 x i32> %2, <32 x i32> %3)
  %6 = tail call <64 x i32> @llvm.hexagon.V6.vmpy.qf32.qf16.128B(<32 x i32> %vina, <32 x i32> %4)
  %7 = tail call <64 x i32> @llvm.hexagon.V6.vmpy.qf32.hf.128B(<32 x i32> %vinb, <32 x i32> %4)
  %8 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %6)
  %9 = tail call <32 x i32> @llvm.hexagon.V6.vsub.qf32.mix.128B(<32 x i32> %8, <32 x i32> %5)
  %10 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %7)
  %11 = tail call <32 x i32> @llvm.hexagon.V6.vsub.qf32.128B(<32 x i32> %10, <32 x i32> %5)
  %12 = tail call <64 x i32> @llvm.hexagon.V6.vcombine.128B(<32 x i32> %11, <32 x i32> %9)
  ret <64 x i32> %12
}

declare <32 x i32> @llvm.hexagon.V6.lvsplath.128B(i32) #1
declare <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32) #1
declare <32 x i32> @llvm.hexagon.V6.vadd.qf16.mix.128B(<32 x i32>, <32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vadd.qf32.mix.128B(<32 x i32>, <32 x i32>) #1
declare <64 x i32> @llvm.hexagon.V6.vmpy.qf32.qf16.128B(<32 x i32>, <32 x i32>) #1
declare <64 x i32> @llvm.hexagon.V6.vmpy.qf32.hf.128B(<32 x i32>, <32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vsub.qf32.mix.128B(<32 x i32>, <32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vsub.qf32.128B(<32 x i32>, <32 x i32>) #1
declare <64 x i32> @llvm.hexagon.V6.vcombine.128B(<32 x i32>, <32 x i32>) #1
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { mustprogress nofree nosync nounwind willreturn memory(none) "approx-func-fp-math"="true" "frame-pointer"="all" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="-long-calls" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(none) }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
