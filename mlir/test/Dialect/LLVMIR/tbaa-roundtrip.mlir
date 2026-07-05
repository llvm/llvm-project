// RUN: mlir-opt %s | mlir-opt | FileCheck %s

#tbaa_root_0 = #llvm.tbaa_root<id = "Simple C/C++ TBAA">
#tbaa_root_1 = #llvm.tbaa_root<id = "Other language TBAA">
#tbaa_root_2 = #llvm.tbaa_root
#tbaa_type_desc_0 = #llvm.tbaa_type_desc<id = "omnipotent char", members = {<#tbaa_root_0, 0>}>
#tbaa_tag_0 = #llvm.tbaa_tag<access_type = #tbaa_type_desc_0, base_type = #tbaa_type_desc_0, offset = 0>
#tbaa_type_desc_1 = #llvm.tbaa_type_desc<id = "long long", members = {<#tbaa_type_desc_0, 0>}>
#tbaa_type_desc_2 = #llvm.tbaa_type_desc<id = "int", members = {<#tbaa_type_desc_0, 0>}>
#tbaa_type_desc_3 = #llvm.tbaa_type_desc<id = "agg2_t", members = {<#tbaa_type_desc_1, 0>, <#tbaa_type_desc_1, 8>}>
#tbaa_type_desc_4 = #llvm.tbaa_type_desc<id = "agg1_t", members = {<#tbaa_type_desc_2, 0>, <#tbaa_type_desc_2, 4>}>
#tbaa_tag_2 = #llvm.tbaa_tag<access_type = #tbaa_type_desc_1, base_type = #tbaa_type_desc_3, offset = 8>
#tbaa_tag_3 = #llvm.tbaa_tag<access_type = #tbaa_type_desc_2, base_type = #tbaa_type_desc_4, offset = 0>
#tbaa_type_desc_5 = #llvm.tbaa_type_desc<id = "omnipotent char", members = {<#tbaa_root_1, 0>}>
#tbaa_type_desc_6 = #llvm.tbaa_type_desc<id = "omnipotent char", members = {<#tbaa_root_2, 0>}>
#tbaa_tag_4 = #llvm.tbaa_tag<access_type = #tbaa_type_desc_6, base_type = #tbaa_type_desc_6, offset = 0>
#tbaa_tag_1 = #llvm.tbaa_tag<access_type = #tbaa_type_desc_5, base_type = #tbaa_type_desc_5, offset = 0>

// CHECK-DAG: #[[$ROOT_0:.*]] = #llvm.tbaa_root<id = "Simple C/C++ TBAA">
// CHECK-DAG: #[[$ROOT_1:.*]] = #llvm.tbaa_root<id = "Other language TBAA">
// CHECK-DAG: #[[$ROOT_2:.*]] = #llvm.tbaa_root
// CHECK-NOT: <{{.*}}>
// CHECK-DAG: #[[$DESC_0:.*]] = #llvm.tbaa_type_desc<id = "omnipotent char", members = {<#[[$ROOT_0]], 0>}>
// CHECK-DAG: #[[$DESC_1:.*]] = #llvm.tbaa_type_desc<id = "long long", members = {<#[[$DESC_0]], 0>}>
// CHECK-DAG: #[[$DESC_2:.*]] = #llvm.tbaa_type_desc<id = "int", members = {<#[[$DESC_0]], 0>}>
// CHECK-DAG: #[[$DESC_3:.*]] = #llvm.tbaa_type_desc<id = "agg2_t", members = {<#[[$DESC_1]], 0>, <#[[$DESC_1]], 8>}>
// CHECK-DAG: #[[$DESC_4:.*]] = #llvm.tbaa_type_desc<id = "agg1_t", members = {<#[[$DESC_2]], 0>, <#[[$DESC_2]], 4>}>
// CHECK-DAG: #[[$DESC_5:.*]] = #llvm.tbaa_type_desc<id = "omnipotent char", members = {<#[[$ROOT_1]], 0>}>
// CHECK-DAG: #[[$DESC_6:.*]] = #llvm.tbaa_type_desc<id = "omnipotent char", members = {<#[[$ROOT_2]], 0>}>
// CHECK-DAG: #[[$TAG_0:.*]] = #llvm.tbaa_tag<base_type = #[[$DESC_0]], access_type = #[[$DESC_0]], offset = 0>
// CHECK-DAG: #[[$TAG_1:.*]] = #llvm.tbaa_tag<base_type = #[[$DESC_5]], access_type = #[[$DESC_5]], offset = 0>
// CHECK-DAG: #[[$TAG_2:.*]] = #llvm.tbaa_tag<base_type = #[[$DESC_3]], access_type = #[[$DESC_1]], offset = 8>
// CHECK-DAG: #[[$TAG_3:.*]] = #llvm.tbaa_tag<base_type = #[[$DESC_4]], access_type = #[[$DESC_2]], offset = 0>
// CHECK-DAG: #[[$TAG_4:.*]] = #llvm.tbaa_tag<base_type = #[[$DESC_6]], access_type = #[[$DESC_6]], offset = 0>

llvm.func @tbaa1(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
  %0 = llvm.mlir.constant(1 : i8) : i8
  llvm.store %0, %arg0 {tbaa = [#tbaa_tag_0]} : i8, !llvm.ptr
  llvm.store %0, %arg1 {tbaa = [#tbaa_tag_1]} : i8, !llvm.ptr
  llvm.return
}

// CHECK:           llvm.func @tbaa1(%[[VAL_0:.*]]: !llvm.ptr, %[[VAL_1:.*]]: !llvm.ptr) {
// CHECK:             %[[VAL_2:.*]] = llvm.mlir.constant(1 : i8) : i8
// CHECK:             llvm.store %[[VAL_2]], %[[VAL_0]] {tbaa = [#[[$TAG_0]]]} : i8, !llvm.ptr
// CHECK:             llvm.store %[[VAL_2]], %[[VAL_1]] {tbaa = [#[[$TAG_1]]]} : i8, !llvm.ptr
// CHECK:             llvm.return
// CHECK:           }

llvm.func @tbaa2(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
  %0 = llvm.mlir.constant(0 : i32) : i32
  %1 = llvm.mlir.constant(1 : i32) : i32
  %2 = llvm.getelementptr inbounds %arg1[%0, 1] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"struct.agg2_t", (i64, i64)>
  %3 = llvm.load %2 {tbaa = [#tbaa_tag_2]} : !llvm.ptr -> i64
  %4 = llvm.trunc %3 : i64 to i32
  %5 = llvm.getelementptr inbounds %arg0[%0, 0] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"struct.agg1_t", (i32, i32)>
  llvm.store %4, %5 {tbaa = [#tbaa_tag_3]} : i32, !llvm.ptr
  llvm.return
}

// CHECK:           llvm.func @tbaa2(%[[VAL_0:.*]]: !llvm.ptr, %[[VAL_1:.*]]: !llvm.ptr) {
// CHECK:             %[[VAL_2:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:             %[[VAL_3:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:             %[[VAL_4:.*]] = llvm.getelementptr inbounds %[[VAL_1]]{{\[}}%[[VAL_2]], 1] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"struct.agg2_t", (i64, i64)>
// CHECK:             %[[VAL_5:.*]] = llvm.load %[[VAL_4]] {tbaa = [#[[$TAG_2]]]} : !llvm.ptr -> i64
// CHECK:             %[[VAL_6:.*]] = llvm.trunc %[[VAL_5]] : i64 to i32
// CHECK:             %[[VAL_7:.*]] = llvm.getelementptr inbounds %[[VAL_0]]{{\[}}%[[VAL_2]], 0] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"struct.agg1_t", (i32, i32)>
// CHECK:             llvm.store %[[VAL_6]], %[[VAL_7]] {tbaa = [#[[$TAG_3]]]} : i32, !llvm.ptr
// CHECK:             llvm.return
// CHECK:           }

llvm.func @tbaa3(%arg0: !llvm.ptr) {
  %0 = llvm.mlir.constant(1 : i8) : i8
  llvm.store %0, %arg0 {tbaa = [#tbaa_tag_0, #tbaa_tag_1]} : i8, !llvm.ptr
  llvm.return
}

// CHECK:           llvm.func @tbaa3(%[[VAL_0:.*]]: !llvm.ptr) {
// CHECK:             %[[VAL_1:.*]] = llvm.mlir.constant(1 : i8) : i8
// CHECK:             llvm.store %[[VAL_1]], %[[VAL_0]] {tbaa = [#[[$TAG_0]], #[[$TAG_1]]]} : i8, !llvm.ptr
// CHECK:             llvm.return
// CHECK:           }

llvm.func @tbaa4(%arg0: !llvm.ptr) {
  %0 = llvm.mlir.constant(1 : i8) : i8
  llvm.store %0, %arg0 {tbaa = [#tbaa_tag_4]} : i8, !llvm.ptr
  llvm.return
}

// CHECK:           llvm.func @tbaa4(%[[VAL_0:.*]]: !llvm.ptr) {
// CHECK:             %[[VAL_1:.*]] = llvm.mlir.constant(1 : i8) : i8
// CHECK:             llvm.store %[[VAL_1]], %[[VAL_0]] {tbaa = [#[[$TAG_4]]]} : i8, !llvm.ptr
// CHECK:             llvm.return
// CHECK:           }
