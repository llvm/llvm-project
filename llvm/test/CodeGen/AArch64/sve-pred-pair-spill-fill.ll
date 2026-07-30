; RUN: llc < %s | FileCheck %s

; Derived from 
; #include <arm_sve.h>

; void g();

; svboolx2_t f0(int64_t i, int64_t n) {
;     svboolx2_t r = svwhilelt_b16_x2(i, n);
;     g();
;     return r;
; }

; svboolx2_t f1(svcount_t n) {
;     svboolx2_t r = svpext_lane_c8_x2(n, 1);
;     g();
;     return r;
; }
; 
; Check that predicate register pairs are spilled/filled without an ICE in the backend.

target triple = "aarch64-unknown-linux"

define <vscale x 32 x i1> @f0(i64 %i, i64 %n) #0 {
entry:
  %0 = tail call { <vscale x 8 x i1>, <vscale x 8 x i1> } @llvm.aarch64.sve.whilelt.x2.nxv8i1(i64 %i, i64 %n)
  %1 = extractvalue { <vscale x 8 x i1>, <vscale x 8 x i1> } %0, 0
  %2 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> %1)
  %3 = tail call <vscale x 32 x i1> @llvm.vector.insert.nxv32i1.nxv16i1(<vscale x 32 x i1> poison, <vscale x 16 x i1> %2, i64 0)
  %4 = extractvalue { <vscale x 8 x i1>, <vscale x 8 x i1> } %0, 1
  %5 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> %4)
  %6 = tail call <vscale x 32 x i1> @llvm.vector.insert.nxv32i1.nxv16i1(<vscale x 32 x i1> %3, <vscale x 16 x i1> %5, i64 16)
  tail call void @g()
  ret <vscale x 32 x i1> %6
}
; CHECK-LABEL: f0:
; CHECK: whilelt { p0.h, p1.h }
; CHECK: str p0, [sp, #6, mul vl]
; CHECK: str p1, [sp, #7, mul vl]
; CHECK: ldr p0, [sp, #6, mul vl]
; CHECK: ldr p1, [sp, #7, mul vl]

define <vscale x 32 x i1> @f1(target("aarch64.svcount") %n) #0 {
entry:
  %0 = tail call { <vscale x 16 x i1>, <vscale x 16 x i1> } @llvm.aarch64.sve.pext.x2.nxv16i1(target("aarch64.svcount") %n, i32 1)
  %1 = extractvalue { <vscale x 16 x i1>, <vscale x 16 x i1> } %0, 0
  %2 = tail call <vscale x 32 x i1> @llvm.vector.insert.nxv32i1.nxv16i1(<vscale x 32 x i1> poison, <vscale x 16 x i1> %1, i64 0)
  %3 = extractvalue { <vscale x 16 x i1>, <vscale x 16 x i1> } %0, 1
  %4 = tail call <vscale x 32 x i1> @llvm.vector.insert.nxv32i1.nxv16i1(<vscale x 32 x i1> %2, <vscale x 16 x i1> %3, i64 16)
  tail call void @g()
  ret <vscale x 32 x i1> %4
}

; CHECK-LABEL: f1:
; CHECK: pext { p0.b, p1.b }
; CHECK: str p0, [sp, #6, mul vl]
; CHECK: str p1, [sp, #7, mul vl]
; CHECK: ldr p0, [sp, #6, mul vl]
; CHECK: ldr p1, [sp, #7, mul vl]

declare void @g(...)
declare { <vscale x 8 x i1>, <vscale x 8 x i1> } @llvm.aarch64.sve.whilelt.x2.nxv8i1(i64, i64)
declare <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1>)
declare <vscale x 32 x i1> @llvm.vector.insert.nxv32i1.nxv16i1(<vscale x 32 x i1>, <vscale x 16 x i1>, i64 immarg)
declare { <vscale x 16 x i1>, <vscale x 16 x i1> } @llvm.aarch64.sve.pext.x2.nxv16i1(target("aarch64.svcount"), i32 immarg) #1

attributes #0 = { nounwind "target-features"="+sve,+sve2,+sve2p1" }
