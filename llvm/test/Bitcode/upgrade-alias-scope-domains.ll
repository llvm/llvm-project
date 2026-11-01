; RUN: llvm-as -disable-verify < %s | llvm-dis | FileCheck %s

define void @test(ptr %p, ptr %q) {
; CHECK-LABEL: define void @test(
; CHECK: call void @llvm.experimental.noalias.scope.decl(metadata [[SELF_SCOPES:![0-9]+]])
; CHECK: load i8, ptr %p, align 1, !alias.scope [[SELF_SCOPES]], !noalias [[NAMED_SCOPES:![0-9]+]]
; CHECK: store i8 %v, ptr %q, align 1, !alias.scope [[NAMED_SCOPES]], !noalias [[SELF_SCOPES]]
  call void @llvm.experimental.noalias.scope.decl(metadata !0)
  %v = load i8, ptr %p, !alias.scope !0, !noalias !3
  store i8 %v, ptr %q, !alias.scope !3, !noalias !0
  ret void
}

define void @shares_the_domain(ptr %p, ptr %q) {
; CHECK-LABEL: define void @shares_the_domain(
; CHECK: load i8, ptr %p, align 1, !alias.scope [[SELF_SCOPES]], !noalias [[NAMED_SCOPES]]
  %v = load i8, ptr %p, !alias.scope !0, !noalias !3
  ret void
}

declare void @llvm.experimental.noalias.scope.decl(metadata)

!0 = !{!1}
!1 = distinct !{!1, !2, !"self-referential scope"}
!2 = distinct !{!2, !"self-referential domain"}
!3 = !{!4}
!4 = !{!"named scope", !5}
!5 = !{!"named domain"}
!6 = !{!7}
!7 = distinct !{!"same name", !8}
!8 = distinct !{!"same domain name"}
!9 = !{!10}
!10 = distinct !{!"same name", !11}
!11 = distinct !{!"same domain name"}

define void @distinct_domains_sharing_a_name(ptr %p, ptr %q) {
; CHECK-LABEL: define void @distinct_domains_sharing_a_name(
; CHECK: load i8, ptr %p, align 1, !alias.scope [[SCOPES_A:![0-9]+]]
; CHECK: store i8 %v, ptr %q, align 1, !noalias [[SCOPES_B:![0-9]+]]
  %v = load i8, ptr %p, !alias.scope !6
  store i8 %v, ptr %q, !noalias !9
  ret void
}

; CHECK-DAG: [[SELF_SCOPES]] = !{[[SELF_SCOPE:![0-9]+]]}
; CHECK-DAG: [[SELF_SCOPE]] = distinct !{[[SELF_SCOPE]], [[SELF_DOMAIN:![0-9]+]], !"self-referential scope"}
; CHECK-DAG: [[SELF_DOMAIN]] = distinct !{[[SELF_DOMAIN]], i1 false, !"self-referential domain"}
; CHECK-DAG: [[NAMED_SCOPES]] = !{[[NAMED_SCOPE:![0-9]+]]}
; CHECK-DAG: [[NAMED_SCOPE]] = !{!"named scope", [[NAMED_DOMAIN:![0-9]+]]}
; CHECK-DAG: [[NAMED_DOMAIN]] = !{!"named domain", i1 false}
; CHECK-DAG: [[SCOPES_A]] = !{[[SCOPE_A:![0-9]+]]}
; CHECK-DAG: [[SCOPE_A]] = distinct !{!"same name", [[DOMAIN_A:![0-9]+]]}
; CHECK-DAG: [[DOMAIN_A]] = distinct !{!"same domain name", i1 false}
; CHECK-DAG: [[SCOPES_B]] = !{[[SCOPE_B:![0-9]+]]}
; CHECK-DAG: [[SCOPE_B]] = distinct !{!"same name", [[DOMAIN_B:![0-9]+]]}
; CHECK-DAG: [[DOMAIN_B]] = distinct !{!"same domain name", i1 false}
