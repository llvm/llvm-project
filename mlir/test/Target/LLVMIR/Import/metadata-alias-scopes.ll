; RUN: mlir-translate -import-llvm -split-input-file %s | FileCheck %s

; CHECK: llvm.metadata @__llvm_global_metadata {
; CHECK:   llvm.alias_scope_domain @[[DOMAIN:.*]] {description = "The domain"}
; CHECK:   llvm.alias_scope @[[$SCOPE0:.*]] {description = "The first scope", domain = @[[DOMAIN]]}
; CHECK:   llvm.alias_scope @[[$SCOPE1:.*]] {domain = @[[DOMAIN]]}
; CHECK:   llvm.alias_scope @[[$SCOPE2:.*]] {domain = @[[DOMAIN]]}
; CHECK: }

; CHECK-LABEL: llvm.func @alias_scope
define void @alias_scope(ptr %arg1) {
  ; CHECK: llvm.load
  ; CHECK-SAME:  alias_scopes = [@__llvm_global_metadata::@[[$SCOPE0]]]
  ; CHECK-SAME:  noalias_scopes = [@__llvm_global_metadata::@[[$SCOPE1]], @__llvm_global_metadata::@[[$SCOPE2]]]
  %1 = load i32, ptr %arg1, !alias.scope !4, !noalias !7
  ; CHECK: llvm.load
  ; CHECK-SAME:  alias_scopes = [@__llvm_global_metadata::@[[$SCOPE1]]]
  ; CHECK-SAME:  noalias_scopes = [@__llvm_global_metadata::@[[$SCOPE0]], @__llvm_global_metadata::@[[$SCOPE2]]]
  %2 = load i32, ptr %arg1, !alias.scope !5, !noalias !8
  ; CHECK: llvm.load
  ; CHECK-SAME:  alias_scopes = [@__llvm_global_metadata::@[[$SCOPE2]]]
  ; CHECK-SAME:  noalias_scopes = [@__llvm_global_metadata::@[[$SCOPE0]], @__llvm_global_metadata::@[[$SCOPE1]]]
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

; CHECK: llvm.metadata @__llvm_global_metadata {
; CHECK:   llvm.alias_scope_domain @[[DOMAIN0:.*]] {description = "The domain"}
; CHECK:   llvm.alias_scope @[[$SCOPE0:.*]] {domain = @[[DOMAIN0]]}
; CHECK:   llvm.alias_scope_domain @[[DOMAIN1:.*]]
; CHECK:   llvm.alias_scope @[[$SCOPE1:.*]] {domain = @[[DOMAIN1]]}
; CHECK: }

; CHECK-LABEL: llvm.func @two_domains
define void @two_domains(ptr %arg1) {
  ; CHECK: llvm.load
  ; CHECK-SAME:  alias_scopes = [@__llvm_global_metadata::@[[$SCOPE0]]]
  ; CHECK-SAME:  noalias_scopes = [@__llvm_global_metadata::@[[$SCOPE1]]]
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

; CHECK: llvm.metadata @__llvm_global_metadata {
; CHECK:   llvm.alias_scope_domain @[[DOMAIN:.*]] {description = "The domain"}
; CHECK:   llvm.alias_scope @[[$SCOPE:.*]] {domain = @[[DOMAIN]]}
; CHECK: }

; CHECK-LABEL: llvm.func @supported_ops
define void @supported_ops(ptr %arg1, float %arg2, i32 %arg3, i32 %arg4) {
  ; CHECK: llvm.intr.experimental.noalias.scope.decl @__llvm_global_metadata::@[[$SCOPE]]
  call void @llvm.experimental.noalias.scope.decl(metadata !2)
  ; CHECK: llvm.load {{.*}}alias_scopes = [@__llvm_global_metadata::@[[$SCOPE]]]
  %1 = load i32, ptr %arg1, !alias.scope !2
  ; CHECK: llvm.store {{.*}}alias_scopes = [@__llvm_global_metadata::@[[$SCOPE]]]
  store i32 %1, ptr %arg1, !alias.scope !2
  ; CHECK: llvm.atomicrmw {{.*}}alias_scopes = [@__llvm_global_metadata::@[[$SCOPE]]]
  %2 = atomicrmw fmax ptr %arg1, float %arg2 acquire, !alias.scope !2
  ; CHECK: llvm.cmpxchg {{.*}}alias_scopes = [@__llvm_global_metadata::@[[$SCOPE]]]
  %3 = cmpxchg ptr %arg1, i32 %arg3, i32 %arg4 monotonic seq_cst, !alias.scope !2
  ; CHECK: "llvm.intr.memcpy"{{.*}}alias_scopes = [@__llvm_global_metadata::@[[$SCOPE]]]
  call void @llvm.memcpy.p0.p0.i32(ptr %arg1, ptr %arg1, i32 4, i1 false), !alias.scope !2
  ; CHECK: "llvm.intr.memset"{{.*}}alias_scopes = [@__llvm_global_metadata::@[[$SCOPE]]]
  call void @llvm.memset.p0.i32(ptr %arg1, i8 42, i32 4, i1 false), !alias.scope !2
  ret void
}

declare void @llvm.experimental.noalias.scope.decl(metadata)
declare void @llvm.memcpy.p0.p0.i32(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i32, i1 immarg)
declare void @llvm.memset.p0.i32(ptr nocapture writeonly, i8, i32, i1 immarg)

!0 = distinct !{!0, !"The domain"}
!1 = !{!1, !0}
!2 = !{!1}
