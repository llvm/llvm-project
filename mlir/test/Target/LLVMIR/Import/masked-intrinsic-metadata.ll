; RUN: mlir-translate -import-llvm %s | FileCheck %s

; CHECK-DAG: #[[$AG:.+]] = #llvm.access_group<id = {{.*}}>
; CHECK-DAG: #[[DOMAIN:.+]] = #llvm.alias_scope_domain<id = {{.*}}, description = "domain">
; CHECK-DAG: #[[TBAA_ROOT:.+]] = #llvm.tbaa_root<id = "Simple C/C++ TBAA">
; CHECK-DAG: #[[$SCOPE:.+]] = #llvm.alias_scope<id = {{.*}}, domain = #[[DOMAIN]], description = "scope">
; CHECK-DAG: #[[$NOALIAS:.+]] = #llvm.alias_scope<id = {{.*}}, domain = #[[DOMAIN]], description = "noalias">
; CHECK-DAG: #[[TBAA_CHAR:.+]] = #llvm.tbaa_type_desc<id = "omnipotent char", members = {<#[[TBAA_ROOT]], 0>}>
; CHECK-DAG: #[[TBAA_INT:.+]] = #llvm.tbaa_type_desc<id = "int", members = {<#[[TBAA_CHAR]], 0>}>
; CHECK-DAG: #[[$TBAA_TAG:.+]] = #llvm.tbaa_tag<base_type = #[[TBAA_INT]], access_type = #[[TBAA_INT]], offset = 0>

; CHECK-LABEL: @masked_load_store_metadata
define void @masked_load_store_metadata(ptr %ptr, <7 x i1> %mask, <7 x float> %val) {
  ; CHECK: llvm.intr.masked.load
  ; CHECK-SAME: access_groups = [#[[$AG]]]
  ; CHECK-SAME: alias_scopes = [#[[$SCOPE]]]
  ; CHECK-SAME: alignment = 4 : i64
  ; CHECK-SAME: noalias_scopes = [#[[$NOALIAS]]]
  ; CHECK-SAME: tbaa = [#[[$TBAA_TAG]]]
  %1 = call <7 x float> @llvm.masked.load.v7f32.p0(ptr align 4 %ptr, <7 x i1> %mask, <7 x float> poison), !tbaa !0, !llvm.access.group !4, !alias.scope !5, !noalias !7
  ; CHECK: llvm.intr.masked.store
  ; CHECK-SAME: access_groups = [#[[$AG]]]
  ; CHECK-SAME: alias_scopes = [#[[$SCOPE]]]
  ; CHECK-SAME: alignment = 4 : i64
  ; CHECK-SAME: noalias_scopes = [#[[$NOALIAS]]]
  ; CHECK-SAME: tbaa = [#[[$TBAA_TAG]]]
  call void @llvm.masked.store.v7f32.p0(<7 x float> %val, ptr align 4 %ptr, <7 x i1> %mask), !tbaa !0, !llvm.access.group !4, !alias.scope !5, !noalias !7
  ret void
}

; CHECK-LABEL: @masked_gather_scatter_metadata
define void @masked_gather_scatter_metadata(<7 x ptr> %ptrs, <7 x i1> %mask, <7 x float> %val) {
  ; CHECK: llvm.intr.masked.gather
  ; CHECK-SAME: access_groups = [#[[$AG]]]
  ; CHECK-SAME: alias_scopes = [#[[$SCOPE]]]
  ; CHECK-SAME: alignment = 4 : i64
  ; CHECK-SAME: noalias_scopes = [#[[$NOALIAS]]]
  ; CHECK-SAME: tbaa = [#[[$TBAA_TAG]]]
  %1 = call <7 x float> @llvm.masked.gather.v7f32.v7p0(<7 x ptr> align 4 %ptrs, <7 x i1> %mask, <7 x float> poison), !tbaa !0, !llvm.access.group !4, !alias.scope !5, !noalias !7
  ; CHECK: llvm.intr.masked.scatter
  ; CHECK-SAME: access_groups = [#[[$AG]]]
  ; CHECK-SAME: alias_scopes = [#[[$SCOPE]]]
  ; CHECK-SAME: alignment = 4 : i64
  ; CHECK-SAME: noalias_scopes = [#[[$NOALIAS]]]
  ; CHECK-SAME: tbaa = [#[[$TBAA_TAG]]]
  call void @llvm.masked.scatter.v7f32.v7p0(<7 x float> %val, <7 x ptr> align 4 %ptrs, <7 x i1> %mask), !tbaa !0, !llvm.access.group !4, !alias.scope !5, !noalias !7
  ret void
}

declare <7 x float> @llvm.masked.load.v7f32.p0(ptr, <7 x i1>, <7 x float>)
declare void @llvm.masked.store.v7f32.p0(<7 x float>, ptr, <7 x i1>)
declare <7 x float> @llvm.masked.gather.v7f32.v7p0(<7 x ptr>, <7 x i1>, <7 x float>)
declare void @llvm.masked.scatter.v7f32.v7p0(<7 x float>, <7 x ptr>, <7 x i1>)

!0 = !{!1, !1, i64 0}
!1 = !{!"int", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
!4 = distinct !{}
!5 = !{!6}
!6 = distinct !{!6, !9, !"scope"}
!7 = !{!8}
!8 = distinct !{!8, !9, !"noalias"}
!9 = distinct !{!9, !"domain"}
