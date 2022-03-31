; RUN: opt -S -passes=sroa %s | FileCheck %s

; SROA should correctly offset `!tbaa.struct` metadata

%struct.Wishart = type { double, i32 }
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* writeonly, i8* readonly, i64, i1 immarg)
declare double @subcall(double %g, i32 %m)

define double @bar(%struct.Wishart* %wishart) {
  %tmp = alloca %struct.Wishart, align 8
  %tmpaddr = bitcast %struct.Wishart* %tmp to i8*
  %waddr = bitcast %struct.Wishart* %wishart to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %tmpaddr, i8* align 8 %waddr, i64 16, i1 false), !tbaa.struct !2
  %gamma = getelementptr inbounds %struct.Wishart, %struct.Wishart* %tmp, i32 0, i32 0
  %lg = load double, double* %gamma, align 8, !tbaa !4
  %m = getelementptr inbounds %struct.Wishart, %struct.Wishart* %tmp, i32 0, i32 1
  %lm = load i32, i32* %m, align 8, !tbaa !8
  %call = call double @subcall(double %lg, i32 %lm)
  ret double %call
}

!2 = !{i64 0, i64 8, !3, i64 8, i64 4, !7}
!3 = !{!4, !4, i64 0}
!4 = !{!"double", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
!7 = !{!8, !8, i64 0}
!8 = !{!"int", !5, i64 0}

; CHECK: define double @bar(%struct.Wishart* %wishart) {
; CHECK-NEXT:   %tmp.sroa.3 = alloca [4 x i8], align 4
; CHECK-NEXT:   %tmp.sroa.0.0.waddr.sroa_idx = getelementptr inbounds %struct.Wishart, %struct.Wishart* %wishart, i64 0, i32 0
; CHECK-NEXT:   %tmp.sroa.0.0.copyload = load double, double* %tmp.sroa.0.0.waddr.sroa_idx, align 8, !tbaa.struct !0
; CHECK-NEXT:   %tmp.sroa.2.0.waddr.sroa_idx1 = getelementptr inbounds %struct.Wishart, %struct.Wishart* %wishart, i64 0, i32 1
; CHECK-NEXT:   %tmp.sroa.2.0.copyload = load i32, i32* %tmp.sroa.2.0.waddr.sroa_idx1, align 8, !tbaa.struct !7
; CHECK-NEXT:   %tmp.sroa.3.0.waddr.sroa_raw_cast = bitcast %struct.Wishart* %wishart to i8*
; CHECK-NEXT:   %tmp.sroa.3.0.waddr.sroa_raw_idx = getelementptr inbounds i8, i8* %tmp.sroa.3.0.waddr.sroa_raw_cast, i64 12
; CHECK-NEXT:   %[[sroa_idx:.+]] = getelementptr inbounds [4 x i8], [4 x i8]* %tmp.sroa.3, i64 0, i64 0
; CHECK-NEXT:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %[[sroa_idx]], i8* align 4 %tmp.sroa.3.0.waddr.sroa_raw_idx, i64 4, i1 false), !tbaa.struct !8
; CHECK-NEXT:   %call = call double @subcall(double %tmp.sroa.0.0.copyload, i32 %tmp.sroa.2.0.copyload)
; CHECK-NEXT:   ret double %call
; CHECK-NEXT: }

; CHECK: !0 = !{i64 0, i64 8, !1, i64 8, i64 4, !5}
; CHECK: !1 = !{!2, !2, i64 0}
; CHECK: !2 = !{!"double", !{{[0-9]+}}, i64 0}

; CHECK: !5 = !{!6, !6, i64 0}
; CHECK: !6 = !{!"int", !{{[0-9]+}}, i64 0}
; CHECK: !7 = !{i64 0, i64 4, !5}
; CHECK: !8 = !{}
