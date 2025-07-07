; RUN: llc -mtriple=hexagon < %s | FileCheck %s
; RUN: llc -mtriple=hexagon -verify-machineinstrs=true < %s | FileCheck %s
; Testing bitreverse load intrinsics:
;   Q6_bitrev_load_update_D(inputLR, pDelay, nConvLength);
;   Q6_bitrev_load_update_W(inputLR, pDelay, nConvLength);
;   Q6_bitrev_load_update_H(inputLR, pDelay, nConvLength);
;   Q6_bitrev_load_update_UH(inputLR, pDelay, nConvLength);
;   Q6_bitrev_load_update_UB(inputLR, pDelay, nConvLength);
;   Q6_bitrev_load_update_B(inputLR, pDelay, nConvLength);
; producing these instructions:
;   r3:2 = memd(r0++m0:brev)
;   r1 = memw(r0++m0:brev)
;   r1 = memh(r0++m0:brev)
;   r1 = memuh(r0++m0:brev)
;   r1 = memub(r0++m0:brev)
;   r1 = memb(r0++m0:brev)

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon-unknown--elf"

; CHECK: @call_brev_ldd
define ptr @call_brev_ldd(ptr %ptr, i64 %dst, i32 %mod) local_unnamed_addr #0 {
entry:
; CHECK: = memd(r{{[0-9]*}}++m{{[0-1]}}:brev)
  %0 = tail call { i64, ptr } @llvm.hexagon.L2.loadrd.pbr(ptr %ptr, i32 %mod)
  %1 = extractvalue { i64, ptr } %0, 1
  ret ptr %1
}

; CHECK: @call_brev_ldw
define ptr @call_brev_ldw(ptr %ptr, i32 %dst, i32 %mod) local_unnamed_addr #0 {
entry:
; CHECK: = memw(r{{[0-9]*}}++m{{[0-1]}}:brev)
  %0 = tail call { i32, ptr } @llvm.hexagon.L2.loadri.pbr(ptr %ptr, i32 %mod)
  %1 = extractvalue { i32, ptr } %0, 1
  ret ptr %1
}

; CHECK: @call_brev_ldh
define ptr @call_brev_ldh(ptr %ptr, i16 signext %dst, i32 %mod) local_unnamed_addr #0 {
entry:
; CHECK: = memh(r{{[0-9]*}}++m{{[0-1]}}:brev)
  %0 = tail call { i32, ptr } @llvm.hexagon.L2.loadrh.pbr(ptr %ptr, i32 %mod)
  %1 = extractvalue { i32, ptr } %0, 1
  ret ptr %1
}

; CHECK: @call_brev_lduh
define ptr @call_brev_lduh(ptr %ptr, i16 zeroext %dst, i32 %mod) local_unnamed_addr #0 {
entry:
; CHECK: = memuh(r{{[0-9]*}}++m{{[0-1]}}:brev)
  %0 = tail call { i32, ptr } @llvm.hexagon.L2.loadruh.pbr(ptr %ptr, i32 %mod)
  %1 = extractvalue { i32, ptr } %0, 1
  ret ptr %1
}

; CHECK: @call_brev_ldb
define ptr @call_brev_ldb(ptr %ptr, i8 signext %dst, i32 %mod) local_unnamed_addr #0 {
entry:
; CHECK: = memb(r{{[0-9]*}}++m{{[0-1]}}:brev)
  %0 = tail call { i32, ptr } @llvm.hexagon.L2.loadrb.pbr(ptr %ptr, i32 %mod)
  %1 = extractvalue { i32, ptr } %0, 1
  ret ptr %1
}

; Function Attrs: nounwind readonly
; CHECK: @call_brev_ldub
define ptr @call_brev_ldub(ptr %ptr, i8 zeroext %dst, i32 %mod) local_unnamed_addr #0 {
entry:
; CHECK: = memub(r{{[0-9]*}}++m{{[0-1]}}:brev)
  %0 = tail call { i32, ptr } @llvm.hexagon.L2.loadrub.pbr(ptr %ptr, i32 %mod)
  %1 = extractvalue { i32, ptr } %0, 1
  ret ptr %1
}

declare { i64, ptr } @llvm.hexagon.L2.loadrd.pbr(ptr, i32) #1
declare { i32, ptr } @llvm.hexagon.L2.loadri.pbr(ptr, i32) #1
declare { i32, ptr } @llvm.hexagon.L2.loadrh.pbr(ptr, i32) #1
declare { i32, ptr } @llvm.hexagon.L2.loadruh.pbr(ptr, i32) #1
declare { i32, ptr } @llvm.hexagon.L2.loadrb.pbr(ptr, i32) #1
declare { i32, ptr } @llvm.hexagon.L2.loadrub.pbr(ptr, i32) #1

attributes #0 = { nounwind readonly "target-cpu"="hexagonv60" }
attributes #1 = { nounwind readonly }
