; RUN: opt < %s -S -passes=memcpyopt | FileCheck --match-full-lines %s

; Alias scopes are merged by taking the intersection of domains, then the union of the scopes within those domains
define i8 @test(i8 %input) {
  %tmp = alloca i8
  %dst = alloca i8
  %src = alloca i8
; CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %dst, i8* align 8 %src, i64 1, i1 false), !alias.scope ![[SCOPE:[0-9]+]]
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %src), !noalias !4
  store i8 %input, i8* %src
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %tmp, i8* align 8 %src, i64 1, i1 false), !alias.scope !0
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %src), !noalias !4
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %dst, i8* align 8 %tmp, i64 1, i1 false), !alias.scope !4
  %ret_value = load i8, i8* %dst
  ret i8 %ret_value
}

; Merged scope contains "callee0: %a" and "callee0 : %b"
; CHECK-DAG: ![[CALLEE0_A:[0-9]+]] = distinct !{!{{[0-9]+}}, !{{[0-9]+}}, !"callee0: %a"}
; CHECK-DAG: ![[CALLEE0_B:[0-9]+]] = distinct !{!{{[0-9]+}}, !{{[0-9]+}}, !"callee0: %b"}
; CHECK-DAG: ![[SCOPE]] = !{![[CALLEE0_A]], ![[CALLEE0_B]]}

declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture)
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture)
declare void @llvm.memcpy.p0i8.p0i8.i64(i8*, i8*, i64, i1)

!0 = !{!1, !7}
!1 = distinct !{!1, !3, !"callee0: %a"}
!2 = distinct !{!2, !3, !"callee0: %b"}
!3 = distinct !{!3, !"callee0"}

!4 = !{!2, !5}
!5 = distinct !{!5, !6, !"callee1: %a"}
!6 = distinct !{!6, !"callee1"}

!7 = distinct !{!7, !8, !"callee2: %a"}
!8 = distinct !{!8, !"callee2"}
