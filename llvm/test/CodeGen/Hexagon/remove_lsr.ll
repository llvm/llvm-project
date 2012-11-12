; Test fix for PR-13709.
; RUN: llc -march=hexagon -mcpu=hexagonv4 < %s | FileCheck %s
; CHECK: foo
; CHECK-NOT: lsr(r{{[0-9]+}}:{{[0-9]+}}, #32)
; CHECK-NOT: lsr(r{{[0-9]+}}:{{[0-9]+}}, #32)

; Convert the sequence
; r17:16 = lsr(r11:10, #32)
; .. = r16
; into
; r17:16 = lsr(r11:10, #32)
; .. = r11
; This makes the lsr instruction dead and it gets removed subsequently
; by a dead code removal pass.

%union.vect64 = type { i64 }
%union.vect32 = type { i32 }

define void @foo(%union.vect64* nocapture %sss_extracted_bit_rx_data_ptr,
 %union.vect32* nocapture %s_even, %union.vect32* nocapture %s_odd,
 i8* nocapture %scr_s_even_code_ptr, i8* nocapture %scr_s_odd_code_ptr)
 nounwind {
entry:
  %scevgep = getelementptr %union.vect64* %sss_extracted_bit_rx_data_ptr, i32 1
  %scevgep28 = getelementptr %union.vect32* %s_odd, i32 1
  %scevgep32 = getelementptr %union.vect32* %s_even, i32 1
  %scevgep36 = getelementptr i8* %scr_s_odd_code_ptr, i32 1
  %scevgep39 = getelementptr i8* %scr_s_even_code_ptr, i32 1
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %lsr.iv42 = phi i32 [ %lsr.iv.next, %for.body ], [ 2, %entry ]
  %lsr.iv40 = phi i8* [ %scevgep41, %for.body ], [ %scevgep39, %entry ]
  %lsr.iv37 = phi i8* [ %scevgep38, %for.body ], [ %scevgep36, %entry ]
  %lsr.iv33 = phi %union.vect32* [ %scevgep34, %for.body ], [ %scevgep32, %entry ]
  %lsr.iv29 = phi %union.vect32* [ %scevgep30, %for.body ], [ %scevgep28, %entry ]
  %lsr.iv = phi %union.vect64* [ %scevgep26, %for.body ], [ %scevgep, %entry ]
  %predicate_1.023 = phi i8 [ undef, %entry ], [ %10, %for.body ]
  %predicate.022 = phi i8 [ undef, %entry ], [ %9, %for.body ]
  %val.021 = phi i64 [ undef, %entry ], [ %srcval, %for.body ]
  %lsr.iv3335 = bitcast %union.vect32* %lsr.iv33 to i32*
  %lsr.iv2931 = bitcast %union.vect32* %lsr.iv29 to i32*
  %lsr.iv27 = bitcast %union.vect64* %lsr.iv to i64*
  %0 = tail call i64 @llvm.hexagon.A2.vsubhs(i64 0, i64 %val.021)
  %conv3 = sext i8 %predicate.022 to i32
  %1 = trunc i64 %val.021 to i32
  %2 = trunc i64 %0 to i32
  %3 = tail call i32 @llvm.hexagon.C2.mux(i32 %conv3, i32 %1, i32 %2)
  store i32 %3, i32* %lsr.iv3335, align 4, !tbaa !0
  %conv8 = sext i8 %predicate_1.023 to i32
  %4 = lshr i64 %val.021, 32
  %5 = trunc i64 %4 to i32
  %6 = lshr i64 %0, 32
  %7 = trunc i64 %6 to i32
  %8 = tail call i32 @llvm.hexagon.C2.mux(i32 %conv8, i32 %5, i32 %7)
  store i32 %8, i32* %lsr.iv2931, align 4, !tbaa !0
  %srcval = load i64* %lsr.iv27, align 8
  %9 = load i8* %lsr.iv40, align 1, !tbaa !1
  %10 = load i8* %lsr.iv37, align 1, !tbaa !1
  %lftr.wideiv = trunc i32 %lsr.iv42 to i8
  %exitcond = icmp eq i8 %lftr.wideiv, 32
  %scevgep26 = getelementptr %union.vect64* %lsr.iv, i32 1
  %scevgep30 = getelementptr %union.vect32* %lsr.iv29, i32 1
  %scevgep34 = getelementptr %union.vect32* %lsr.iv33, i32 1
  %scevgep38 = getelementptr i8* %lsr.iv37, i32 1
  %scevgep41 = getelementptr i8* %lsr.iv40, i32 1
  %lsr.iv.next = add i32 %lsr.iv42, 1
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

declare i64 @llvm.hexagon.A2.vsubhs(i64, i64) nounwind readnone

declare i32 @llvm.hexagon.C2.mux(i32, i32, i32) nounwind readnone

!0 = metadata !{metadata !"long", metadata !1}
!1 = metadata !{metadata !"omnipotent char", metadata !2}
!2 = metadata !{metadata !"Simple C/C++ TBAA", null}
