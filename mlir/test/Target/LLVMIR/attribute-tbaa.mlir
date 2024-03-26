// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

#tbaa_root_0 = #ptr.tbaa_root<id = "Simple C/C++ TBAA">
#tbaa_type_desc_1 = #ptr.tbaa_type_desc<id = "omnipotent char", members = {<#tbaa_root_0, 0>}>
#tbaa_type_desc_2 = #ptr.tbaa_type_desc<id = "long long", members = {<#tbaa_type_desc_1, 0>}>
#tbaa_type_desc_3 = #ptr.tbaa_type_desc<id = "agg2_t", members = {<#tbaa_type_desc_2, 0>, <#tbaa_type_desc_2, 8>}>
#tbaa_tag_4 = #ptr.tbaa_tag<access_type = #tbaa_type_desc_2, base_type = #tbaa_type_desc_3, offset = 8>
#tbaa_type_desc_5 = #ptr.tbaa_type_desc<id = "int", members = {<#tbaa_type_desc_1, 0>}>
#tbaa_type_desc_6 = #ptr.tbaa_type_desc<id = "agg1_t", members = {<#tbaa_type_desc_5, 0>, <#tbaa_type_desc_5, 4>}>
#tbaa_tag_7 = #ptr.tbaa_tag<access_type = #tbaa_type_desc_5, base_type = #tbaa_type_desc_6, offset = 0, constant = true>

llvm.func @tbaa2(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
  %0 = llvm.mlir.constant(0 : i32) : i32
  %1 = llvm.mlir.constant(1 : i32) : i32
  %2 = llvm.getelementptr inbounds %arg1[%0, 1] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"struct.agg2_t", (i64, i64)>
  // CHECK: load i64, ptr %{{.*}},{{.*}}!tbaa ![[LTAG:[0-9]*]]
  %3 = ptr.load %2 {tbaa = [#tbaa_tag_4]} : !llvm.ptr -> i64
  %4 = llvm.trunc %3 : i64 to i32
  %5 = llvm.getelementptr inbounds %arg0[%0, 0] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"struct.agg1_t", (i32, i32)>
  // CHECK: store i32 %{{.*}}, ptr %{{.*}},{{.*}}!tbaa ![[STAG:[0-9]*]]
  ptr.store %4, %5 {tbaa = [#tbaa_tag_7]} : i32, !llvm.ptr
  llvm.return
}

// CHECK-DAG: ![[LTAG]] = !{![[AGG2T:[0-9]*]], ![[I64T:[0-9]*]], i64 8}
// CHECK-DAG: ![[AGG2T]] = !{!"agg2_t", ![[I64T]], i64 0, ![[I64T]], i64 8}
// CHECK-DAG: ![[I64T]] = !{!"long long", ![[CHART:[0-9]*]], i64 0}
// CHECK-DAG: ![[CHART]] = !{!"omnipotent char", ![[ROOT:[0-9]*]], i64 0}
// CHECK-DAG: ![[ROOT]] = !{!"Simple C/C++ TBAA"}
// CHECK-DAG: ![[STAG]] = !{![[AGG1T:[0-9]*]], ![[I32T:[0-9]*]], i64 0, i64 1}
// CHECK-DAG: ![[AGG1T]] = !{!"agg1_t", ![[I32T]], i64 0, ![[I32T]], i64 4}
// CHECK-DAG: ![[I32T]] = !{!"int", ![[CHART]], i64 0}

// -----

// Verify that the MDNode's created for the access tags are not uniqued
// before they are finalized. In the process of creating the MDNodes for
// the tag operations we used to produce incomplete MDNodes like:
//   #tbaa_tag_4 => !{!null, !null, i64 0}
//   #tbaa_tag_7 => !{!null, !null, i64 0}
// This caused the two tags to map to the same incomplete MDNode due to
// uniquing. To prevent this, we have to use temporary MDNodes
// instead of !null's.

#tbaa_root_0 = #ptr.tbaa_root<id = "Simple C/C++ TBAA">
#tbaa_type_desc_1 = #ptr.tbaa_type_desc<id = "omnipotent char", members = {<#tbaa_root_0, 0>}>
#tbaa_type_desc_2 = #ptr.tbaa_type_desc<id = "float", members = {<#tbaa_type_desc_1, 0>}>
#tbaa_type_desc_3 = #ptr.tbaa_type_desc<id = "agg2_t", members = {<#tbaa_type_desc_2, 0>, <#tbaa_type_desc_2, 4>}>
#tbaa_tag_4 = #ptr.tbaa_tag<access_type = #tbaa_type_desc_2, base_type = #tbaa_type_desc_3, offset = 0>
#tbaa_type_desc_5 = #ptr.tbaa_type_desc<id = "int", members = {<#tbaa_type_desc_1, 0>}>
#tbaa_type_desc_6 = #ptr.tbaa_type_desc<id = "agg1_t", members = {<#tbaa_type_desc_5, 0>, <#tbaa_type_desc_5, 4>}>
#tbaa_tag_7 = #ptr.tbaa_tag<access_type = #tbaa_type_desc_5, base_type = #tbaa_type_desc_6, offset = 0>


llvm.func @foo(%arg0: !llvm.ptr)
llvm.func @tbaa2(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
  %0 = llvm.mlir.constant(0 : i32) : i32
  %1 = llvm.mlir.constant(1 : i32) : i32
  %2 = llvm.getelementptr inbounds %arg1[%0, 0] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"struct.agg2_t", (f32, f32)>
  // CHECK: load float, ptr %{{.*}},{{.*}}!tbaa ![[LTAG:[0-9]*]]
  %3 = ptr.load %2 {tbaa = [#tbaa_tag_4]} : !llvm.ptr -> f32
  %4 = llvm.fptosi %3 : f32 to i32
  %5 = llvm.getelementptr inbounds %arg0[%0, 0] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"struct.agg1_t", (i32, i32)>
  // CHECK: store i32 %{{.*}}, ptr %{{.*}},{{.*}}!tbaa ![[STAG:[0-9]*]]
  ptr.store %4, %5 {tbaa = [#tbaa_tag_7]} : i32, !llvm.ptr
  // CHECK: atomicrmw add ptr %{{.*}}, i32 %{{.*}} !tbaa ![[STAG]]
  %6 = ptr.atomicrmw add %5, %4 monotonic {tbaa = [#tbaa_tag_7]} : !llvm.ptr, i32
  // CHECK: cmpxchg ptr %{{.*}}, i32 %{{.*}}, i32 %{{.*}} !tbaa ![[STAG]]
  %7, %8 = ptr.cmpxchg %5, %6, %4 acq_rel monotonic {tbaa = [#tbaa_tag_7]} : !llvm.ptr, i32
  %9 = llvm.mlir.constant(42 : i8) : i8
  // CHECK: llvm.memcpy{{.*}} !tbaa ![[STAG]]
  "llvm.intr.memcpy"(%arg1, %arg1, %0) <{isVolatile = false}> {tbaa = [#tbaa_tag_7]} : (!llvm.ptr, !llvm.ptr, i32) -> ()
  // CHECK: llvm.memset{{.*}} !tbaa ![[STAG]]
  "llvm.intr.memset"(%arg1, %9, %0) <{isVolatile = false}> {tbaa = [#tbaa_tag_7]} : (!llvm.ptr, i8, i32) -> ()
  // CHECK: call void @foo({{.*}} !tbaa ![[STAG]]
  llvm.call @foo(%arg1) {tbaa = [#tbaa_tag_7]} : (!llvm.ptr) -> ()
  llvm.return
}


// CHECK-DAG: ![[LTAG]] = !{![[AGG2T:[0-9]*]], ![[F32T:[0-9]*]], i64 0}
// CHECK-DAG: ![[AGG2T]] = !{!"agg2_t", ![[F32T]], i64 0, ![[F32T]], i64 4}
// CHECK-DAG: ![[I64T]] = !{!"float", ![[CHART:[0-9]*]], i64 0}
// CHECK-DAG: ![[CHART]] = !{!"omnipotent char", ![[ROOT:[0-9]*]], i64 0}
// CHECK-DAG: ![[ROOT]] = !{!"Simple C/C++ TBAA"}
// CHECK-DAG: ![[STAG]] = !{![[AGG1T:[0-9]*]], ![[I32T:[0-9]*]], i64 0}
// CHECK-DAG: ![[AGG1T]] = !{!"agg1_t", ![[I32T]], i64 0, ![[I32T]], i64 4}
// CHECK-DAG: ![[I32T]] = !{!"int", ![[CHART]], i64 0}
