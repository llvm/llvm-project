; RUN: opt -S -mtriple=x86_64-pc-linux-gnu -mcpu=generic -slp-vectorizer -pass-remarks-output=%t < %s | FileCheck %s
; RUN: FileCheck --input-file=%t --check-prefix=YAML %s

define void @vsub2_test(i32* %pin1, i32* %pin2, i32* %pout) #0 {
  br label %1

  %idx.04 = phi i32 [ 0, %0 ], [ %8, %1 ]
  %po.03 = phi i32* [ %pout, %0 ], [ %7, %1 ]
  %ptmpi2.02 = phi i32* [ %pin2, %0 ], [ %4, %1 ]
  %ptmpi1.01 = phi i32* [ %pin1, %0 ], [ %2, %1 ]
  %2 = getelementptr inbounds i32, i32* %ptmpi1.01, i64 1
  %3 = load i32, i32* %ptmpi1.01, align 4, !tbaa !1
  %4 = getelementptr inbounds i32, i32* %ptmpi2.02, i64 1
  %5 = load i32, i32* %ptmpi2.02, align 4, !tbaa !1
  %6 = sub nsw i32 %3, %5
  %7 = getelementptr inbounds i32, i32* %po.03, i64 1
 ; CHECK-NOT: <{{[0-9]+}} x i32>
 ; YAML:      Pass:            slp-vectorizer
 ; YAML-NEXT: Name:            NotBeneficial
 ; YAML-NEXT: Function:        vsub2_test
 ; YAML-NEXT: Args:
 ; YAML-NEXT:   - String:          'List vectorization was possible but not beneficial with cost '
 ; YAML-NEXT:   - Cost:            '0'
 ; YAML-NEXT:   - String:          ' >= '
 ; YAML-NEXT:   - Treshold:        '0'
  store i32 %6, i32* %po.03, align 4, !tbaa !1
  %8 = add nuw nsw i32 %idx.04, 1
  %exitcond = icmp eq i32 %8, 64
  br i1 %exitcond, label %9, label %1, !llvm.loop !5

  ret void
}

!llvm.ident = !{!0}

!0 = !{!"clang version 3.8.0-2ubuntu4 (tags/RELEASE_380/final)"}
!1 = !{!2, !2, i64 0}
!2 = !{!"int", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
!5 = distinct !{!5, !6, !7}
!6 = !{!"llvm.loop.vectorize.width", i32 1}
!7 = !{!"llvm.loop.interleave.count", i32 1}
