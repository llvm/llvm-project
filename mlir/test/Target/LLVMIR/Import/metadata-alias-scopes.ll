; RUN: mlir-translate -import-llvm -split-input-file %s | FileCheck %s

; CHECK: #[[DOMAIN:.*]] = #ptr.alias_scope_domain<id = {{.*}}, description = "The domain">
; CHECK: #[[$SCOPE0:.*]] = #ptr.alias_scope<id = {{.*}}, domain = #[[DOMAIN]], description = "The first scope">
; CHECK: #[[$SCOPE1:.*]] = #ptr.alias_scope<id = {{.*}}, domain = #[[DOMAIN]]>
; CHECK: #[[$SCOPE2:.*]] = #ptr.alias_scope<id = {{.*}}, domain = #[[DOMAIN]]>

; CHECK-LABEL: llvm.func @alias_scope
define void @alias_scope(ptr %arg1) {
  ; CHECK: ptr.load
  ; CHECK-SAME:  alias_scopes = [#[[$SCOPE0]]]
  ; CHECK-SAME:  noalias_scopes = [#[[$SCOPE1]], #[[$SCOPE2]]]
  %1 = load i32, ptr %arg1, !alias.scope !4, !noalias !7
  ; CHECK: ptr.load
  ; CHECK-SAME:  alias_scopes = [#[[$SCOPE1]]]
  ; CHECK-SAME:  noalias_scopes = [#[[$SCOPE0]], #[[$SCOPE2]]]
  %2 = load i32, ptr %arg1, !alias.scope !5, !noalias !8
  ; CHECK: ptr.load
  ; CHECK-SAME:  alias_scopes = [#[[$SCOPE2]]]
  ; CHECK-SAME:  noalias_scopes = [#[[$SCOPE0]], #[[$SCOPE1]]]
  %3 = load i32, ptr %arg1, !alias.scope !6, !noalias !9
  ret void
}

!0 = distinct !{!0, !"The domain"}
!1 = distinct !{!1, !0, !"The first scope"}
!2 = distinct !{!2, !0}
!3 = distinct !{!3, !0}
!4 = !{!1}
!5 = !{!2}
!6 = !{!3}
!7 = !{!2, !3}
!8 = !{!1, !3}
!9 = !{!1, !2}

; // -----

; CHECK: #[[DOMAIN0:.*]] = #ptr.alias_scope_domain<id = {{.*}}, description = "The domain">
; CHECK: #[[DOMAIN1:.*]] = #ptr.alias_scope_domain<id = {{.*}}>
; CHECK: #[[$SCOPE0:.*]] = #ptr.alias_scope<id = {{.*}}, domain = #[[DOMAIN0]]>
; CHECK: #[[$SCOPE1:.*]] = #ptr.alias_scope<id = {{.*}}, domain = #[[DOMAIN1]]>

; CHECK-LABEL: llvm.func @two_domains
define void @two_domains(ptr %arg1) {
  ; CHECK: ptr.load
  ; CHECK-SAME:  alias_scopes = [#[[$SCOPE0]]]
  ; CHECK-SAME:  noalias_scopes = [#[[$SCOPE1]]]
  %1 = load i32, ptr %arg1, !alias.scope !4, !noalias !5
  ret void
}

!0 = distinct !{!0, !"The domain"}
!1 = distinct !{!1}
!2 = !{!2, !0}
!3 = !{!3, !1}
!4 = !{!2}
!5 = !{!3}

; // -----

; CHECK: #[[DOMAIN:.*]] = #ptr.alias_scope_domain<id = {{.*}}, description = "The domain">
; CHECK: #[[$SCOPE:.*]] = #ptr.alias_scope<id = {{.*}}, domain = #[[DOMAIN]]>

; CHECK-LABEL: llvm.func @supported_ops
define void @supported_ops(ptr %arg1, float %arg2, i32 %arg3, i32 %arg4) {
  ; CHECK: llvm.intr.experimental.noalias.scope.decl #[[$SCOPE]]
  call void @llvm.experimental.noalias.scope.decl(metadata !2)
  ; CHECK: ptr.load {{.*}}alias_scopes = [#[[$SCOPE]]]
  %1 = load i32, ptr %arg1, !alias.scope !2
  ; CHECK: ptr.store {{.*}}alias_scopes = [#[[$SCOPE]]]
  store i32 %1, ptr %arg1, !alias.scope !2
  ; CHECK: ptr.atomicrmw {{.*}}alias_scopes = [#[[$SCOPE]]]
  %2 = atomicrmw fmax ptr %arg1, float %arg2 acquire, !alias.scope !2
  ; CHECK: ptr.cmpxchg {{.*}}alias_scopes = [#[[$SCOPE]]]
  %3 = cmpxchg ptr %arg1, i32 %arg3, i32 %arg4 monotonic seq_cst, !alias.scope !2
  ; CHECK: "llvm.intr.memcpy"{{.*}}alias_scopes = [#[[$SCOPE]]]
  call void @llvm.memcpy.p0.p0.i32(ptr %arg1, ptr %arg1, i32 4, i1 false), !alias.scope !2
  ; CHECK: "llvm.intr.memset"{{.*}}alias_scopes = [#[[$SCOPE]]]
  call void @llvm.memset.p0.i32(ptr %arg1, i8 42, i32 4, i1 false), !alias.scope !2
  ; CHECK: llvm.call{{.*}}alias_scopes = [#[[$SCOPE]]]
  call void @foo(ptr %arg1), !alias.scope !2
  ; CHECK: llvm.call{{.*}}noalias_scopes = [#[[$SCOPE]]]
  call void @foo(ptr %arg1), !noalias !2
  ret void
}

declare void @llvm.experimental.noalias.scope.decl(metadata)
declare void @llvm.memcpy.p0.p0.i32(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i32, i1 immarg)
declare void @llvm.memset.p0.i32(ptr nocapture writeonly, i8, i32, i1 immarg)
declare void @foo(ptr %arg1)

!0 = distinct !{!0, !"The domain"}
!1 = !{!1, !0}
!2 = !{!1}
