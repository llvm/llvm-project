; RUN: llc -march=hexagon < %s | FileCheck %s
; RUN: llc -march=hexagon -verify-machineinstrs=true < %s | FileCheck %s
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
define i64* @call_brev_ldd(i64* %ptr, i64 %dst, i32 %mod) local_unnamed_addr #0 {
entry:
  %0 = bitcast i64* %ptr to i8*
; CHECK: = memd(r{{[0-9]*}}++m{{[0-1]}}:brev)
  %1 = tail call { i64, i8* } @llvm.hexagon.L2.loadrd.pbr(i8* %0, i32 %mod)
  %2 = extractvalue { i64, i8* } %1, 1
  %3 = bitcast i8* %2 to i64*
  ret i64* %3
}

; CHECK: @call_brev_ldw
define i32* @call_brev_ldw(i32* %ptr, i32 %dst, i32 %mod) local_unnamed_addr #0 {
entry:
  %0 = bitcast i32* %ptr to i8*
; CHECK: = memw(r{{[0-9]*}}++m{{[0-1]}}:brev)
  %1 = tail call { i32, i8* } @llvm.hexagon.L2.loadri.pbr(i8* %0, i32 %mod)
  %2 = extractvalue { i32, i8* } %1, 1
  %3 = bitcast i8* %2 to i32*
  ret i32* %3
}

; CHECK: @call_brev_ldh
define i16* @call_brev_ldh(i16* %ptr, i16 signext %dst, i32 %mod) local_unnamed_addr #0 {
entry:
  %0 = bitcast i16* %ptr to i8*
; CHECK: = memh(r{{[0-9]*}}++m{{[0-1]}}:brev)
  %1 = tail call { i32, i8* } @llvm.hexagon.L2.loadrh.pbr(i8* %0, i32 %mod)
  %2 = extractvalue { i32, i8* } %1, 1
  %3 = bitcast i8* %2 to i16*
  ret i16* %3
}

; CHECK: @call_brev_lduh
define i16* @call_brev_lduh(i16* %ptr, i16 zeroext %dst, i32 %mod) local_unnamed_addr #0 {
entry:
  %0 = bitcast i16* %ptr to i8*
; CHECK: = memuh(r{{[0-9]*}}++m{{[0-1]}}:brev)
  %1 = tail call { i32, i8* } @llvm.hexagon.L2.loadruh.pbr(i8* %0, i32 %mod)
  %2 = extractvalue { i32, i8* } %1, 1
  %3 = bitcast i8* %2 to i16*
  ret i16* %3
}

; CHECK: @call_brev_ldb
define i8* @call_brev_ldb(i8* %ptr, i8 signext %dst, i32 %mod) local_unnamed_addr #0 {
entry:
; CHECK: = memb(r{{[0-9]*}}++m{{[0-1]}}:brev)
  %0 = tail call { i32, i8* } @llvm.hexagon.L2.loadrb.pbr(i8* %ptr, i32 %mod)
  %1 = extractvalue { i32, i8* } %0, 1
  ret i8* %1
}

; Function Attrs: nounwind readonly
; CHECK: @call_brev_ldub
define i8* @call_brev_ldub(i8* %ptr, i8 zeroext %dst, i32 %mod) local_unnamed_addr #0 {
entry:
; CHECK: = memub(r{{[0-9]*}}++m{{[0-1]}}:brev)
  %0 = tail call { i32, i8* } @llvm.hexagon.L2.loadrub.pbr(i8* %ptr, i32 %mod)
  %1 = extractvalue { i32, i8* } %0, 1
  ret i8* %1
}

declare { i64, i8* } @llvm.hexagon.L2.loadrd.pbr(i8*, i32) #1
declare { i32, i8* } @llvm.hexagon.L2.loadri.pbr(i8*, i32) #1
declare { i32, i8* } @llvm.hexagon.L2.loadrh.pbr(i8*, i32) #1
declare { i32, i8* } @llvm.hexagon.L2.loadruh.pbr(i8*, i32) #1
declare { i32, i8* } @llvm.hexagon.L2.loadrb.pbr(i8*, i32) #1
declare { i32, i8* } @llvm.hexagon.L2.loadrub.pbr(i8*, i32) #1

attributes #0 = { nounwind readonly "target-cpu"="hexagonv60" }
attributes #1 = { nounwind readonly }
