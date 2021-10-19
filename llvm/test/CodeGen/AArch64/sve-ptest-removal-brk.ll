; RUN: llc -mtriple=aarch64--linux-gnu -mattr=+sve %s -o - | FileCheck %s

; Test that redundant ptest instruction is removed when using a flag setting brk

define i32 @brkpb(<vscale x 16 x i1> %pg, <vscale x 16 x i1> %a, <vscale x 16 x i1> %b) {
; CHECK-LABEL: brkpb:
; CHECK: brkpbs p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: cset w0, ne
; CHECK-NEXT: ret
  %1 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.brkpb.z.nxv16i1(<vscale x 16 x i1> %pg, <vscale x 16 x i1> %a, <vscale x 16 x i1> %b)
  %2 = tail call i1 @llvm.aarch64.sve.ptest.any.nxv16i1(<vscale x 16 x i1> %pg, <vscale x 16 x i1> %1)
  %conv = zext i1 %2 to i32
  ret i32 %conv
}

define i32 @brkb(<vscale x 16 x i1> %pg, <vscale x 16 x i1> %a) {
; CHECK-LABEL: brkb:
; CHECK: brkbs p0.b, p0/z, p1.b
; CHECK-NEXT: cset w0, ne
; CHECK-NEXT: ret
  %1 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.brkb.z.nxv16i1(<vscale x 16 x i1> %pg, <vscale x 16 x i1> %a)
  %2 = tail call i1 @llvm.aarch64.sve.ptest.any.nxv16i1(<vscale x 16 x i1> %pg, <vscale x 16 x i1> %1)
  %conv = zext i1 %2 to i32
  ret i32 %conv
}

define i32 @brkn(<vscale x 16 x i1> %pg, <vscale x 16 x i1> %a, <vscale x 16 x i1> %b) {
; CHECK-LABEL: brkn:
; CHECK: brkns p2.b, p0/z, p1.b, p2.b
; CHECK-NEXT: cset w0, ne
; CHECK-NEXT: ret
  %1 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.brkn.z.nxv16i1(<vscale x 16 x i1> %pg, <vscale x 16 x i1> %a, <vscale x 16 x i1> %b)
  %2 = tail call i1 @llvm.aarch64.sve.ptest.any.nxv16i1(<vscale x 16 x i1> %pg, <vscale x 16 x i1> %1)
  %conv = zext i1 %2 to i32
  ret i32 %conv
}

; Test that ptest instruction is not removed when using a non-flag setting brk

define i32 @brkpb_neg(<vscale x 16 x i1> %pg, <vscale x 16 x i1> %a, <vscale x 16 x i1> %b) {
; CHECK-LABEL: brkpb_neg:
; CHECK: brkpb p0.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ptest p1, p0.b
; CHECK-NEXT: cset w0, ne
; CHECK-NEXT: ret
  %1 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.brkpb.z.nxv16i1(<vscale x 16 x i1> %pg, <vscale x 16 x i1> %a, <vscale x 16 x i1> %b)
  %2 = tail call i1 @llvm.aarch64.sve.ptest.any.nxv16i1(<vscale x 16 x i1> %a, <vscale x 16 x i1> %1)
  %conv = zext i1 %2 to i32
  ret i32 %conv
}

define i32 @brkb_neg(<vscale x 16 x i1> %pg, <vscale x 16 x i1> %a) {
; CHECK-LABEL: brkb_neg:
; CHECK: brkb p0.b, p0/z, p1.b
; CHECK-NEXT: ptest p1, p0.b
; CHECK-NEXT: cset w0, ne
; CHECK-NEXT: ret
  %1 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.brkb.z.nxv16i1(<vscale x 16 x i1> %pg, <vscale x 16 x i1> %a)
  %2 = tail call i1 @llvm.aarch64.sve.ptest.any.nxv16i1(<vscale x 16 x i1> %a, <vscale x 16 x i1> %1)
  %conv = zext i1 %2 to i32
  ret i32 %conv
}

define i32 @brkn_neg(<vscale x 16 x i1> %pg, <vscale x 16 x i1> %a, <vscale x 16 x i1> %b) {
; CHECK-LABEL: brkn_neg:
; CHECK: brkn p2.b, p0/z, p1.b, p2.b
; CHECK-NEXT: ptest p1, p2.b
; CHECK-NEXT: cset w0, ne
; CHECK-NEXT: ret
  %1 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.brkn.z.nxv16i1(<vscale x 16 x i1> %pg, <vscale x 16 x i1> %a, <vscale x 16 x i1> %b)
  %2 = tail call i1 @llvm.aarch64.sve.ptest.any.nxv16i1(<vscale x 16 x i1> %a, <vscale x 16 x i1> %1)
  %conv = zext i1 %2 to i32
  ret i32 %conv
}

declare <vscale x 16 x i1> @llvm.aarch64.sve.brkpb.z.nxv16i1(<vscale x 16 x i1>, <vscale x 16 x i1>, <vscale x 16 x i1>)
declare <vscale x 16 x i1> @llvm.aarch64.sve.brkb.z.nxv16i1(<vscale x 16 x i1>, <vscale x 16 x i1>)
declare <vscale x 16 x i1> @llvm.aarch64.sve.brkn.z.nxv16i1(<vscale x 16 x i1>, <vscale x 16 x i1>, <vscale x 16 x i1>)
declare i1 @llvm.aarch64.sve.ptest.any.nxv16i1(<vscale x 16 x i1>, <vscale x 16 x i1>)
