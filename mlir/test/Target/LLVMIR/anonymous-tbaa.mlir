// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

#tbaa_root_0 = #llvm.tbaa_root<>
#tbaa_type_desc_1 = #llvm.tbaa_type_desc<id = "omnipotent char", members = {<#tbaa_root_0, 0>}>
#tbaa_type_desc_2 = #llvm.tbaa_type_desc<id = "long long", members = {<#tbaa_type_desc_1, 0>}>
#tbaa_tag_3 = #llvm.tbaa_tag<access_type = #tbaa_type_desc_2, base_type = #tbaa_type_desc_2, offset = 0>

// CHECK: define void @tbaa_anonymous_root(ptr %{{.*}}) {
// CHECK:   %{{.*}} = load i64, ptr %{{.*}}, align 4, !tbaa ![[TAG:[0-9]+]]
// CHECK:   ret void
// CHECK: }
// CHECK: !llvm.module.flags = !{![[FLAGS:[0-9]+]]}
// CHECK: ![[FLAGS]] = !{i32 2, !"Debug Info Version", i32 3}
// CHECK: ![[TAG]] = !{![[TYPE:[0-9]+]], ![[TYPE]], i64 0}
// CHECK: ![[TYPE]] = !{!"long long", ![[BASE:[0-9]+]], i64 0}
// CHECK: ![[BASE]] = !{!"omnipotent char", ![[ROOT:[0-9]+]], i64 0}
// CHECK: ![[ROOT]] = distinct !{![[ROOT]]}
llvm.func @tbaa_anonymous_root(%arg0: !llvm.ptr) {
  %0 = llvm.load %arg0 {tbaa = [#tbaa_tag_3]} : !llvm.ptr -> i64
  llvm.return
}
