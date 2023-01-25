// RUN: mlir-opt %s | mlir-opt | FileCheck %s

module {
  llvm.metadata @__tbaa {
    llvm.tbaa_root @tbaa_root_0 {id = "Simple C/C++ TBAA"}
    llvm.tbaa_tag @tbaa_tag_1 {access_type = @tbaa_root_0, base_type = @tbaa_root_0, offset = 0 : i64}
    llvm.tbaa_root @tbaa_root_2 {id = "Other language TBAA"}
    llvm.tbaa_tag @tbaa_tag_3 {access_type = @tbaa_root_2, base_type = @tbaa_root_2, offset = 0 : i64}
  }
  llvm.func @tbaa1(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    %0 = llvm.mlir.constant(1 : i8) : i8
    llvm.store %0, %arg0 {tbaa = [@__tbaa::@tbaa_tag_1]} : i8, !llvm.ptr
    llvm.store %0, %arg1 {tbaa = [@__tbaa::@tbaa_tag_3]} : i8, !llvm.ptr
    llvm.return
  }
}

// CHECK-LABEL:     llvm.metadata @__tbaa {
// CHECK:             llvm.tbaa_root @tbaa_root_0 {id = "Simple C/C++ TBAA"}
// CHECK:             llvm.tbaa_tag @tbaa_tag_1 {access_type = @tbaa_root_0, base_type = @tbaa_root_0, offset = 0 : i64}
// CHECK:             llvm.tbaa_root @tbaa_root_2 {id = "Other language TBAA"}
// CHECK:             llvm.tbaa_tag @tbaa_tag_3 {access_type = @tbaa_root_2, base_type = @tbaa_root_2, offset = 0 : i64}
// CHECK:           }
// CHECK:           llvm.func @tbaa1(%[[VAL_0:.*]]: !llvm.ptr, %[[VAL_1:.*]]: !llvm.ptr) {
// CHECK:             %[[VAL_2:.*]] = llvm.mlir.constant(1 : i8) : i8
// CHECK:             llvm.store %[[VAL_2]], %[[VAL_0]] {tbaa = [@__tbaa::@tbaa_tag_1]} : i8, !llvm.ptr
// CHECK:             llvm.store %[[VAL_2]], %[[VAL_1]] {tbaa = [@__tbaa::@tbaa_tag_3]} : i8, !llvm.ptr
// CHECK:             llvm.return
// CHECK:           }

module {
  llvm.metadata @__tbaa {
    llvm.tbaa_root @tbaa_root_0 {id = "Simple C/C++ TBAA"}
    llvm.tbaa_type_desc @tbaa_type_desc_1 {id = "omnipotent char", members = {<@tbaa_root_0, 0>}}
    llvm.tbaa_type_desc @tbaa_type_desc_2 {id = "long long", members = {<@tbaa_type_desc_1, 0>}}
    llvm.tbaa_type_desc @tbaa_type_desc_3 {id = "agg2_t", members = {<@tbaa_type_desc_2, 0>, <@tbaa_type_desc_2, 8>}}
    llvm.tbaa_tag @tbaa_tag_4 {access_type = @tbaa_type_desc_2, base_type = @tbaa_type_desc_3, offset = 8 : i64}
    llvm.tbaa_type_desc @tbaa_type_desc_5 {id = "int", members = {<@tbaa_type_desc_1, 0>}}
    llvm.tbaa_type_desc @tbaa_type_desc_6 {id = "agg1_t", members = {<@tbaa_type_desc_5, 0>, <@tbaa_type_desc_5, 4>}}
    llvm.tbaa_tag @tbaa_tag_7 {access_type = @tbaa_type_desc_5, base_type = @tbaa_type_desc_6, offset = 0 : i64}
  }
  llvm.func @tbaa2(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    %0 = llvm.mlir.constant(0 : i32) : i32
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.getelementptr inbounds %arg1[%0, 1] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"struct.agg2_t", (i64, i64)>
    %3 = llvm.load %2 {tbaa = [@__tbaa::@tbaa_tag_4]} : !llvm.ptr -> i64
    %4 = llvm.trunc %3 : i64 to i32
    %5 = llvm.getelementptr inbounds %arg0[%0, 0] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"struct.agg1_t", (i32, i32)>
    llvm.store %4, %5 {tbaa = [@__tbaa::@tbaa_tag_7]} : i32, !llvm.ptr
    llvm.return
  }
}

// CHECK-LABEL:     llvm.metadata @__tbaa {
// CHECK:             llvm.tbaa_root @tbaa_root_0 {id = "Simple C/C++ TBAA"}
// CHECK:             llvm.tbaa_type_desc @tbaa_type_desc_1 {id = "omnipotent char", members = {<@tbaa_root_0, 0>}}
// CHECK:             llvm.tbaa_type_desc @tbaa_type_desc_2 {id = "long long", members = {<@tbaa_type_desc_1, 0>}}
// CHECK:             llvm.tbaa_type_desc @tbaa_type_desc_3 {id = "agg2_t", members = {<@tbaa_type_desc_2, 0>, <@tbaa_type_desc_2, 8>}}
// CHECK:             llvm.tbaa_tag @tbaa_tag_4 {access_type = @tbaa_type_desc_2, base_type = @tbaa_type_desc_3, offset = 8 : i64}
// CHECK:             llvm.tbaa_type_desc @tbaa_type_desc_5 {id = "int", members = {<@tbaa_type_desc_1, 0>}}
// CHECK:             llvm.tbaa_type_desc @tbaa_type_desc_6 {id = "agg1_t", members = {<@tbaa_type_desc_5, 0>, <@tbaa_type_desc_5, 4>}}
// CHECK:             llvm.tbaa_tag @tbaa_tag_7 {access_type = @tbaa_type_desc_5, base_type = @tbaa_type_desc_6, offset = 0 : i64}
// CHECK:           }
// CHECK:           llvm.func @tbaa2(%[[VAL_0:.*]]: !llvm.ptr, %[[VAL_1:.*]]: !llvm.ptr) {
// CHECK:             %[[VAL_2:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:             %[[VAL_3:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:             %[[VAL_4:.*]] = llvm.getelementptr inbounds %[[VAL_1]]{{\[}}%[[VAL_2]], 1] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"struct.agg2_t", (i64, i64)>
// CHECK:             %[[VAL_5:.*]] = llvm.load %[[VAL_4]] {tbaa = [@__tbaa::@tbaa_tag_4]} : !llvm.ptr -> i64
// CHECK:             %[[VAL_6:.*]] = llvm.trunc %[[VAL_5]] : i64 to i32
// CHECK:             %[[VAL_7:.*]] = llvm.getelementptr inbounds %[[VAL_0]]{{\[}}%[[VAL_2]], 0] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"struct.agg1_t", (i32, i32)>
// CHECK:             llvm.store %[[VAL_6]], %[[VAL_7]] {tbaa = [@__tbaa::@tbaa_tag_7]} : i32, !llvm.ptr
// CHECK:             llvm.return
// CHECK:           }

module {
  llvm.metadata @__tbaa {
    llvm.tbaa_root @tbaa_root_0 {id = "Simple C/C++ TBAA"}
    llvm.tbaa_tag @tbaa_tag_1 {access_type = @tbaa_root_0, base_type = @tbaa_root_0, offset = 0 : i64}
    llvm.tbaa_root @tbaa_root_2 {id = "Other language TBAA"}
    llvm.tbaa_tag @tbaa_tag_3 {access_type = @tbaa_root_2, base_type = @tbaa_root_2, offset = 0 : i64}
  }
  llvm.func @tbaa1(%arg0: !llvm.ptr) {
    %0 = llvm.mlir.constant(1 : i8) : i8
    llvm.store %0, %arg0 {tbaa = [@__tbaa::@tbaa_tag_1, @__tbaa::@tbaa_tag_3]} : i8, !llvm.ptr
    llvm.return
  }
}

// CHECK-LABEL:     llvm.metadata @__tbaa {
// CHECK:             llvm.tbaa_root @tbaa_root_0 {id = "Simple C/C++ TBAA"}
// CHECK:             llvm.tbaa_tag @tbaa_tag_1 {access_type = @tbaa_root_0, base_type = @tbaa_root_0, offset = 0 : i64}
// CHECK:             llvm.tbaa_root @tbaa_root_2 {id = "Other language TBAA"}
// CHECK:             llvm.tbaa_tag @tbaa_tag_3 {access_type = @tbaa_root_2, base_type = @tbaa_root_2, offset = 0 : i64}
// CHECK:           }
// CHECK:           llvm.func @tbaa1(%[[VAL_0:.*]]: !llvm.ptr) {
// CHECK:             %[[VAL_1:.*]] = llvm.mlir.constant(1 : i8) : i8
// CHECK:             llvm.store %[[VAL_1]], %[[VAL_0]] {tbaa = [@__tbaa::@tbaa_tag_1, @__tbaa::@tbaa_tag_3]} : i8, !llvm.ptr
// CHECK:             llvm.return
// CHECK:           }
