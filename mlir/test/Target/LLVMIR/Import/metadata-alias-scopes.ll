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
