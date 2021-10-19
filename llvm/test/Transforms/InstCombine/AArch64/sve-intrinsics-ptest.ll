; RUN: opt -S -instcombine < %s | FileCheck %s

target triple = "aarch64-unknown-linux-gnu"

define i1 @ptest_any1(<vscale x 2 x i1> %a) #0 {
; CHECK-LABEL: ptest_any1
; CHECK: %mask = tail call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 0)
; CHECK-NOT: convert
; CHECK-NEXT: %[[OUT:.*]] = call i1 @llvm.aarch64.sve.ptest.any.nxv2i1(<vscale x 2 x i1> %mask, <vscale x 2 x i1> %a)
; CHECK-NEXT: ret i1 %[[OUT]]
  %mask = tail call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 0)
  %1 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1> %mask)
  %2 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1> %a)
  %out = call i1 @llvm.aarch64.sve.ptest.any.nxv16i1(<vscale x 16 x i1> %1, <vscale x 16 x i1> %2)
  ret i1 %out
}

; No transform because the ptest is using differently sized operands.
define i1 @ptest_any2(<vscale x 4 x i1> %a) #0 {
; CHECK-LABEL: ptest_any2
; CHECK: %mask = tail call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 31)
; CHECK-NEXT: %1 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1> %mask)
; CHECK-NEXT: %2 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv4i1(<vscale x 4 x i1> %a)
; CHECK-NEXT: %out = call i1 @llvm.aarch64.sve.ptest.any.nxv16i1(<vscale x 16 x i1> %1, <vscale x 16 x i1> %2)
  %mask = tail call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 31)
  %1 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1> %mask)
  %2 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv4i1(<vscale x 4 x i1> %a)
  %out = call i1 @llvm.aarch64.sve.ptest.any.nxv16i1(<vscale x 16 x i1> %1, <vscale x 16 x i1> %2)
  ret i1 %out
}

define i1 @ptest_first(<vscale x 4 x i1> %a) #0 {
; CHECK-LABEL: ptest_first
; CHECK: %mask = tail call <vscale x 4 x i1> @llvm.aarch64.sve.ptrue.nxv4i1(i32 0)
; CHECK-NOT: convert
; CHECK-NEXT: %[[OUT:.*]] = call i1 @llvm.aarch64.sve.ptest.first.nxv4i1(<vscale x 4 x i1> %mask, <vscale x 4 x i1> %a)
; CHECK-NEXT: ret i1 %[[OUT]]
  %mask = tail call <vscale x 4 x i1> @llvm.aarch64.sve.ptrue.nxv4i1(i32 0)
  %1 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv4i1(<vscale x 4 x i1> %mask)
  %2 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv4i1(<vscale x 4 x i1> %a)
  %out = call i1 @llvm.aarch64.sve.ptest.first.nxv16i1(<vscale x 16 x i1> %1, <vscale x 16 x i1> %2)
  ret i1 %out
}

define i1 @ptest_first_same_ops(<vscale x 2 x i1> %a) #0 {
; CHECK-LABEL: ptest_first_same_ops
; CHECK: %[[OUT:.*]] = call i1 @llvm.aarch64.sve.ptest.first.nxv2i1(<vscale x 2 x i1> %a, <vscale x 2 x i1> %a)
; CHECK-NOT: convert
; CHECK-NEXT: ret i1 %[[OUT]]
  %1 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1> %a)
  %2 = tail call i1 @llvm.aarch64.sve.ptest.first.nxv16i1(<vscale x 16 x i1> %1, <vscale x 16 x i1> %1)
  ret i1 %2
}

define i1 @ptest_last(<vscale x 8 x i1> %a) #0 {
; CHECK-LABEL: ptest_last
; CHECK: %mask = tail call <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32 0)
; CHECK-NOT: convert
; CHECK-NEXT: %[[OUT:.*]] = call i1 @llvm.aarch64.sve.ptest.last.nxv8i1(<vscale x 8 x i1> %mask, <vscale x 8 x i1> %a)
; CHECK-NEXT: ret i1 %[[OUT]]
  %mask = tail call <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32 0)
  %1 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> %mask)
  %2 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> %a)
  %out = call i1 @llvm.aarch64.sve.ptest.last.nxv16i1(<vscale x 16 x i1> %1, <vscale x 16 x i1> %2)
  ret i1 %out
}

declare <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32)
declare <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32)
declare <vscale x 4 x i1> @llvm.aarch64.sve.ptrue.nxv4i1(i32)
declare <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32)

declare i1 @llvm.aarch64.sve.ptest.any.nxv16i1(<vscale x 16 x i1>, <vscale x 16 x i1>)
declare i1 @llvm.aarch64.sve.ptest.first.nxv16i1(<vscale x 16 x i1>, <vscale x 16 x i1>)
declare i1 @llvm.aarch64.sve.ptest.last.nxv16i1(<vscale x 16 x i1>, <vscale x 16 x i1>)

declare <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1>)
declare <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv4i1(<vscale x 4 x i1>)
declare <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1>)

attributes #0 = { "target-features"="+sve" }
