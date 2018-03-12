; RUN: llc -march=hexagon < %s | FileCheck %s

; CHECK-LABEL: f0:
; CHECK: r{{[1-9]+:[0-9]+}} = memd(r{{[0-9]*}}++#{{[0-9]}}:circ(m{{[01]}}))
define i64 @f0(i64* %a0) {
b0:
  %v0 = alloca i64, align 8
  %v1 = getelementptr inbounds i64, i64* %a0, i32 1
  store i64 0, i64* %v0, align 8, !tbaa !0
  %v2 = bitcast i64* %v1 to i8*
  %v3 = bitcast i64* %v0 to i8*
  %v4 = call i8* @llvm.hexagon.circ.ldd(i8* %v2, i8* %v3, i32 150996984, i32 8)
  %v5 = load i64, i64* %v0, align 8, !tbaa !0
  ret i64 %v5
}

; Function Attrs: argmemonly nounwind
declare i8* @llvm.hexagon.circ.ldd(i8*, i8*, i32, i32) #0

; CHECK-LABEL: f1:
; CHECK: r{{[0-9]*}} = memb(r{{[0-9]*}}++#{{[0-9]}}:circ(m{{[01]}}))
define signext i8 @f1(i8* %a0) {
b0:
  %v0 = alloca i8, align 1
  %v1 = getelementptr inbounds i8, i8* %a0, i32 1
  store i8 0, i8* %v0, align 1, !tbaa !4
  %v2 = call i8* @llvm.hexagon.circ.ldb(i8* %v1, i8* %v0, i32 16777471, i32 1)
  %v3 = load i8, i8* %v0, align 1, !tbaa !4
  ret i8 %v3
}

; Function Attrs: argmemonly nounwind
declare i8* @llvm.hexagon.circ.ldb(i8*, i8*, i32, i32) #0

; CHECK-LABEL: f2:
; CHECK: r{{[0-9]*}} = memub(r{{[0-9]*}}++#{{[0-9]}}:circ(m{{[01]}}))
define signext i8 @f2(i8* %a0) {
b0:
  %v0 = alloca i8, align 1
  %v1 = getelementptr inbounds i8, i8* %a0, i32 1
  store i8 0, i8* %v0, align 1, !tbaa !4
  %v2 = call i8* @llvm.hexagon.circ.ldub(i8* %v1, i8* %v0, i32 16777471, i32 1)
  %v3 = load i8, i8* %v0, align 1, !tbaa !4
  ret i8 %v3
}

; Function Attrs: argmemonly nounwind
declare i8* @llvm.hexagon.circ.ldub(i8*, i8*, i32, i32) #0

; CHECK-LABEL: f3:
; CHECK: r{{[0-9]*}} = memh(r{{[0-9]*}}++#{{[0-9]}}:circ(m{{[01]}}))
define signext i16 @f3(i16* %a0) {
b0:
  %v0 = alloca i16, align 2
  %v1 = getelementptr inbounds i16, i16* %a0, i32 1
  store i16 0, i16* %v0, align 2, !tbaa !5
  %v2 = bitcast i16* %v1 to i8*
  %v3 = bitcast i16* %v0 to i8*
  %v4 = call i8* @llvm.hexagon.circ.ldh(i8* %v2, i8* %v3, i32 33554942, i32 2)
  %v5 = load i16, i16* %v0, align 2, !tbaa !5
  ret i16 %v5
}

; Function Attrs: argmemonly nounwind
declare i8* @llvm.hexagon.circ.ldh(i8*, i8*, i32, i32) #0

; CHECK-LABEL: f4:
; CHECK: r{{[0-9]*}} = memuh(r{{[0-9]*}}++#{{[0-9]}}:circ(m{{[01]}}))
define signext i16 @f4(i16* %a0) {
b0:
  %v0 = alloca i16, align 2
  %v1 = getelementptr inbounds i16, i16* %a0, i32 1
  store i16 0, i16* %v0, align 2, !tbaa !5
  %v2 = bitcast i16* %v1 to i8*
  %v3 = bitcast i16* %v0 to i8*
  %v4 = call i8* @llvm.hexagon.circ.lduh(i8* %v2, i8* %v3, i32 33554942, i32 2)
  %v5 = load i16, i16* %v0, align 2, !tbaa !5
  ret i16 %v5
}

; Function Attrs: argmemonly nounwind
declare i8* @llvm.hexagon.circ.lduh(i8*, i8*, i32, i32) #0

; CHECK-LABEL: f5:
; CHECK: r{{[0-9]*}} = memw(r{{[0-9]*}}++#{{[0-9]}}:circ(m{{[01]}}))
define i32 @f5(i32* %a0) {
b0:
  %v0 = alloca i32, align 4
  %v1 = getelementptr inbounds i32, i32* %a0, i32 1
  store i32 0, i32* %v0, align 4, !tbaa !7
  %v2 = bitcast i32* %v1 to i8*
  %v3 = bitcast i32* %v0 to i8*
  %v4 = call i8* @llvm.hexagon.circ.ldw(i8* %v2, i8* %v3, i32 50332668, i32 4)
  %v5 = load i32, i32* %v0, align 4, !tbaa !7
  ret i32 %v5
}

; Function Attrs: argmemonly nounwind
declare i8* @llvm.hexagon.circ.ldw(i8*, i8*, i32, i32) #0

attributes #0 = { argmemonly nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"long long", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
!4 = !{!2, !2, i64 0}
!5 = !{!6, !6, i64 0}
!6 = !{!"short", !2, i64 0}
!7 = !{!8, !8, i64 0}
!8 = !{!"long", !2, i64 0}
