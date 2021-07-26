; REQUIRES: asserts
; RUN: opt -loop-vectorize -scalable-vectorization=on -S -mtriple=aarch64 -mattr=+sve -debug-only=loop-vectorize < %s 2>&1 | FileCheck %s

target triple = "aarch64-unknown-linux-gnu"

; In the test below the PHI instruction:
;  %0 = phi i8* [ %incdec.ptr190, %loop.body ], [ %src, %entry ]
; has multiple uses, i.e.
;  1. As a uniform address for the load, and
;  2. Non-uniform use by the getelementptr + store, which leads to replication.

; CHECK-LABEL:  LV: Checking a loop in "phi_multiple_use"
; CHECK-NOT:    LV: Found new scalar instruction:   %incdec.ptr190 = getelementptr inbounds i8, i8* %0, i64 1
;
; CHECK:        VPlan 'Initial VPlan for VF={vscale x 2},UF>=1' {
; CHECK-NEXT:   loop.body:
; CHECK-NEXT:     WIDEN-INDUCTION %index = phi 0, %index.next
; CHECK-NEXT:     WIDEN-PHI %curchar = phi %curchar.next, %curptr
; CHECK-NEXT:     WIDEN-PHI %0 = phi %incdec.ptr190, %src
; CHECK-NEXT:     WIDEN-GEP Var[Inv] ir<%incdec.ptr190> = getelementptr ir<%0>, ir<1>
; CHECK-NEXT:     WIDEN store ir<%curchar>, ir<%incdec.ptr190>
; CHECK-NEXT:     WIDEN ir<%1> = load ir<%0>
; CHECK-NEXT:     WIDEN ir<%2> = add ir<%1>, ir<1>
; CHECK-NEXT:     WIDEN store ir<%0>, ir<%2>
; CHECK-NEXT:   No successors
; CHECK-NEXT:   }

define void @phi_multiple_use(i8** noalias %curptr, i8* noalias %src, i64 %N) #0 {
; CHECK-LABEL: @phi_multiple_use(
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX1:%.*]] = phi i64 [ 0, %vector.ph ], [ {{.*}}, %vector.body ]
; CHECK-NEXT:    {{.*}} = add i64 [[INDEX1]], 0
; CHECK-NEXT:    [[TMP1:%.*]] = add i64 [[INDEX1]], 0
; CHECK-NEXT:    [[NEXT_GEP:%.*]] = getelementptr i8*, i8** %curptr, i64 [[TMP1]]
; CHECK-NEXT:    [[TMP2:%.*]] = call <vscale x 2 x i64> @llvm.experimental.stepvector.nxv2i64()
; CHECK-NEXT:    [[DOTSPLATINSERT:%.*]] = insertelement <vscale x 2 x i64> poison, i64 [[INDEX1]], i32 0
; CHECK-NEXT:    [[DOTSPLAT:%.*]] = shufflevector <vscale x 2 x i64> [[DOTSPLATINSERT]], <vscale x 2 x i64> poison, <vscale x 2 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP3:%.*]] = add <vscale x 2 x i64> shufflevector (<vscale x 2 x i64> insertelement (<vscale x 2 x i64> poison, i64 0, i32 0), <vscale x 2 x i64> poison, <vscale x 2 x i32> zeroinitializer), [[TMP2]]
; CHECK-NEXT:    [[TMP4:%.*]] = add <vscale x 2 x i64> [[DOTSPLAT]], [[TMP3]]
; CHECK-NEXT:    [[NEXT_GEP6:%.*]] = getelementptr i8, i8* %src, <vscale x 2 x i64> [[TMP4]]
; CHECK-NEXT:    [[TMP5:%.*]] = getelementptr inbounds i8, <vscale x 2 x i8*> [[NEXT_GEP6]], i64 1
; CHECK:         store <vscale x 2 x i8*> [[TMP5]], <vscale x 2 x i8*>*
; CHECK-NEXT:    [[TMP6:%.*]] = extractelement <vscale x 2 x i8*> [[NEXT_GEP6]], i32 0
; CHECK-NEXT:    [[TMP7:%.*]] = getelementptr i8, i8* [[TMP6]], i32 0
; CHECK-NEXT:    [[TMP8:%.*]] = bitcast i8* [[TMP7]] to <vscale x 2 x i8>*
; CHECK-NEXT:    [[WIDE_LOAD:%.*]] = load <vscale x 2 x i8>, <vscale x 2 x i8>* [[TMP8]]
; CHECK-NEXT:    [[TMP9:%.*]] = add <vscale x 2 x i8> [[WIDE_LOAD]],
; CHECK:         store <vscale x 2 x i8> [[TMP9]], <vscale x 2 x i8>*

entry:
  br label %loop.body

loop.body:                                    ; preds = %loop.body, %entry
  %index = phi i64 [ 0, %entry ], [ %index.next, %loop.body ]
  %curchar = phi i8** [ %curchar.next, %loop.body ], [ %curptr, %entry ]
  %0 = phi i8* [ %incdec.ptr190, %loop.body ], [ %src, %entry ]
  %incdec.ptr190 = getelementptr inbounds i8, i8* %0, i64 1
  %curchar.next = getelementptr inbounds i8*, i8** %curchar, i64 1
  store i8* %incdec.ptr190, i8** %curchar, align 8
  %1 = load i8, i8* %0, align 1
  %2 = add i8 %1, 1
  store i8 %2, i8* %0, align 1
  %index.next = add nuw i64 %index, 1
  %3 = icmp ne i64 %index.next, %N
  br i1 %3, label %loop.body, label %exit, !llvm.loop !0

exit:                            ; preds = %loop.body
  ret void
}


attributes #0 = {"target-features"="+sve"}

!0 = distinct !{!0, !1, !2, !3}
!1 = !{!"llvm.loop.interleave.count", i32 1}
!2 = !{!"llvm.loop.vectorize.width", i32 2}
!3 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}
!4 = !{ !5 }
!5 = distinct !{ !5, !6 }
!6 = distinct !{ !7 }
!7 = distinct !{ !7, !6 }
