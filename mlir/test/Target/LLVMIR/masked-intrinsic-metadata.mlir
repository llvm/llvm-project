// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

#alias_scope_domain = #llvm.alias_scope_domain<id = distinct[0]<>, description = "The domain">
#alias_scope1 = #llvm.alias_scope<id = distinct[1]<>, domain = #alias_scope_domain, description = "The first scope">
#alias_scope2 = #llvm.alias_scope<id = distinct[2]<>, domain = #alias_scope_domain, description = "The second scope">
#access_group = #llvm.access_group<id = distinct[3]<>>
#tbaa_root = #llvm.tbaa_root<id = "Simple C/C++ TBAA">
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "int", members = {<#tbaa_root, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc, access_type = #tbaa_type_desc, offset = 0>

// CHECK-LABEL: @masked_load_store_metadata
llvm.func @masked_load_store_metadata(%ptr: !llvm.ptr, %mask: vector<7xi1>) {
  // CHECK: call <7 x float> @llvm.masked.load.v7f32.p0
  // CHECK-SAME: !tbaa ![[$TBAA:[0-9]+]]
  // CHECK-SAME: !alias.scope ![[$SCOPE1:[0-9]+]]
  // CHECK-SAME: !noalias ![[$SCOPE2:[0-9]+]]
  // CHECK-SAME: !llvm.access.group ![[$AG:[0-9]+]]
  %0 = llvm.intr.masked.load %ptr, %mask {
      alignment = 4 : i64,
      access_groups = [#access_group],
      alias_scopes = [#alias_scope1],
      noalias_scopes = [#alias_scope2],
      tbaa = [#tbaa_tag]} : (!llvm.ptr, vector<7xi1>) -> vector<7xf32>
  // CHECK: call void @llvm.masked.store.v7f32.p0
  // CHECK-SAME: !tbaa ![[$TBAA]]
  // CHECK-SAME: !alias.scope ![[$SCOPE1]]
  // CHECK-SAME: !noalias ![[$SCOPE2]]
  // CHECK-SAME: !llvm.access.group ![[$AG]]
  llvm.intr.masked.store %0, %ptr, %mask {
      alignment = 4 : i64,
      access_groups = [#access_group],
      alias_scopes = [#alias_scope1],
      noalias_scopes = [#alias_scope2],
      tbaa = [#tbaa_tag]} : vector<7xf32>, vector<7xi1> into !llvm.ptr
  llvm.return
}

// CHECK-LABEL: @masked_gather_scatter_metadata
llvm.func @masked_gather_scatter_metadata(%ptrs: vector<7 x !llvm.ptr>, %mask: vector<7xi1>) {
  // CHECK: call <7 x float> @llvm.masked.gather.v7f32.v7p0
  // CHECK-SAME: !tbaa ![[$TBAA]]
  // CHECK-SAME: !alias.scope ![[$SCOPE1]]
  // CHECK-SAME: !noalias ![[$SCOPE2]]
  // CHECK-SAME: !llvm.access.group ![[$AG]]
  %0 = llvm.intr.masked.gather %ptrs, %mask {
      alignment = 4 : i64,
      access_groups = [#access_group],
      alias_scopes = [#alias_scope1],
      noalias_scopes = [#alias_scope2],
      tbaa = [#tbaa_tag]} : (vector<7 x !llvm.ptr>, vector<7xi1>) -> vector<7xf32>
  // CHECK: call void @llvm.masked.scatter.v7f32.v7p0
  // CHECK-SAME: !tbaa ![[$TBAA]]
  // CHECK-SAME: !alias.scope ![[$SCOPE1]]
  // CHECK-SAME: !noalias ![[$SCOPE2]]
  // CHECK-SAME: !llvm.access.group ![[$AG]]
  llvm.intr.masked.scatter %0, %ptrs, %mask {
      alignment = 4 : i64,
      access_groups = [#access_group],
      alias_scopes = [#alias_scope1],
      noalias_scopes = [#alias_scope2],
      tbaa = [#tbaa_tag]} : vector<7xf32>, vector<7xi1> into vector<7 x !llvm.ptr>
  llvm.return
}

// CHECK-DAG: ![[$TBAA]] = !{![[TBAA_TYPE:[0-9]+]], ![[TBAA_TYPE]], i64 0}
// CHECK-DAG: ![[TBAA_TYPE]] = !{!"int", ![[TBAA_ROOT:[0-9]+]], i64 0}
// CHECK-DAG: ![[TBAA_ROOT]] = !{!"Simple C/C++ TBAA"}
// CHECK-DAG: ![[$SCOPE1]] = !{![[SCOPE1_DECL:[0-9]+]]}
// CHECK-DAG: ![[SCOPE1_DECL]] = distinct !{![[SCOPE1_DECL]], ![[DOMAIN:[0-9]+]], !"The first scope"}
// CHECK-DAG: ![[DOMAIN]] = distinct !{![[DOMAIN]], !"The domain"}
// CHECK-DAG: ![[$SCOPE2]] = !{![[SCOPE2_DECL:[0-9]+]]}
// CHECK-DAG: ![[SCOPE2_DECL]] = distinct !{![[SCOPE2_DECL]], ![[DOMAIN]], !"The second scope"}
// CHECK-DAG: ![[$AG]] = distinct !{}
