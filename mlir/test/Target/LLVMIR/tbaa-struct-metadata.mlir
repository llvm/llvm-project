// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s
// RUN: mlir-opt %s --verify-roundtrip -o /dev/null

// The export mirror of Import/metadata-tbaa-struct.ll. The body is exactly what
// the importer produces, so the two pin the whole trip.
//
// The tags inside a tbaa_struct are reachable from nowhere else in the module,
// so the exporter has to walk them explicitly; without that the node comes out
// with null tags.

// CHECK: call void @llvm.memcpy.p0.p0.i64(ptr %{{.*}}, ptr %{{.*}}, i64 12, i1 false), !tbaa.struct ![[S:[0-9]+]]
// CHECK: call void @llvm.memmove.p0.p0.i64(ptr %{{.*}}, ptr %{{.*}}, i64 12, i1 false), !tbaa.struct ![[S]]
// CHECK: call void @llvm.memcpy.inline.p0.p0.i64(ptr %{{.*}}, ptr %{{.*}}, i64 12, i1 false), !tbaa.struct ![[S]]
// CHECK: ![[S]] = !{i64 0, i64 4, ![[T0:[0-9]+]], i64 8, i64 4, ![[T1:[0-9]+]]}
// CHECK-DAG: ![[T0]] = !{![[INT:[0-9]+]], ![[INT]], i64 0}
// CHECK-DAG: ![[INT]] = !{!"int", !{{[0-9]+}}, i64 0}
// CHECK-DAG: ![[T1]] = !{![[FLT:[0-9]+]], ![[FLT]], i64 0}
// CHECK-DAG: ![[FLT]] = !{!"float", !{{[0-9]+}}, i64 0}

#tbaa_root = #llvm.tbaa_root<id = "Simple C++ TBAA">
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "omnipotent char", members = {<#tbaa_root, 0>}>
#tbaa_type_desc1 = #llvm.tbaa_type_desc<id = "int", members = {<#tbaa_type_desc, 0>}>
#tbaa_type_desc2 = #llvm.tbaa_type_desc<id = "float", members = {<#tbaa_type_desc, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc1, access_type = #tbaa_type_desc1, offset = 0>
#tbaa_tag1 = #llvm.tbaa_tag<base_type = #tbaa_type_desc2, access_type = #tbaa_type_desc2, offset = 0>
module {
  llvm.func @cp(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    %0 = llvm.mlir.constant(12 : i64) : i64
    "llvm.intr.memcpy"(%arg0, %arg1, %0) <{isVolatile = false, tbaa_struct = [#llvm.tbaa_struct_field<offset = 0, size = 4, tag = #tbaa_tag>, #llvm.tbaa_struct_field<offset = 8, size = 4, tag = #tbaa_tag1>]}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    "llvm.intr.memmove"(%arg0, %arg1, %0) <{isVolatile = false, tbaa_struct = [#llvm.tbaa_struct_field<offset = 0, size = 4, tag = #tbaa_tag>, #llvm.tbaa_struct_field<offset = 8, size = 4, tag = #tbaa_tag1>]}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    "llvm.intr.memcpy.inline"(%arg0, %arg1) <{isVolatile = false, len = 12 : i64, tbaa_struct = [#llvm.tbaa_struct_field<offset = 0, size = 4, tag = #tbaa_tag>, #llvm.tbaa_struct_field<offset = 8, size = 4, tag = #tbaa_tag1>]}> : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }
}
