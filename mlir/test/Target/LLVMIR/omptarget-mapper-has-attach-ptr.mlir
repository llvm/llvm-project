// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Verify that HasAttachPtr controls mapper MEMBER_OF adjustment for a pointee
// storage block reached through an attach pointer. The combined pointee block
// entry must not be made MEMBER_OF the outer object, but member entries inside
// that pointee block keep their inner MEMBER_OF relation, shifted by the
// mapper's current component count. The ATTACH entry remains unmodified.

module attributes {omp.target_triples = ["amdgcn-amd-amdhsa"]} {
  omp.declare_mapper @mapper : !llvm.struct<"S2", (i32, ptr)> {
  ^bb0(%arg0: !llvm.ptr):
    %field = llvm.getelementptr %arg0[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"S2", (i32, ptr)>
    %ptee = llvm.load %field : !llvm.ptr -> !llvm.ptr
    %x = llvm.getelementptr %ptee[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"S1", (i32, i32)>
    %y = llvm.getelementptr %ptee[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"S1", (i32, i32)>
    %map_x = omp.map.info var_ptr(%x : !llvm.ptr, i32)
        map_clauses(tofrom) capture(ByRef) name("x") has_attach_ptr(true) -> !llvm.ptr
    %map_y = omp.map.info var_ptr(%y : !llvm.ptr, i32)
        map_clauses(tofrom) capture(ByRef) name("y") has_attach_ptr(true) -> !llvm.ptr
    %map_ptee = omp.map.info var_ptr(%ptee : !llvm.ptr, !llvm.struct<"S1", (i32, i32)>)
        map_clauses(tofrom) capture(ByRef) members(%map_x, %map_y : [0], [1] : !llvm.ptr, !llvm.ptr) name("s1p") partial_map(true) has_attach_ptr(true) -> !llvm.ptr
    %map_attach = omp.map.info var_ptr(%arg0 : !llvm.ptr, !llvm.struct<"S2", (i32, ptr)>)
        map_clauses(attach, ref_ptr, ref_ptee) capture(ByRef) var_ptr_ptr(%field : !llvm.ptr, !llvm.struct<"S1", (i32, i32)>) name("s1p") -> !llvm.ptr
    omp.declare_mapper.info map_entries(%map_ptee, %map_x, %map_y, %map_attach : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr)
  }

  llvm.func @test(%arg0: !llvm.ptr) {
    %map = omp.map.info var_ptr(%arg0 : !llvm.ptr, !llvm.struct<"S2", (i32, ptr)>)
        map_clauses(tofrom) capture(ByRef) mapper(@mapper) name("s2") -> !llvm.ptr
    omp.target kernel_type(generic) map_entries(%map -> %arg1 : !llvm.ptr) {
      omp.terminator
    }
    llvm.return
  }
}

// CHECK-LABEL: define internal void @.omp_mapper.mapper
// CHECK: %[[PREV:.*]] = call i64 @__tgt_mapper_num_components
// CHECK: %[[SHIFT:.*]] = shl i64 %[[PREV]], 48
// CHECK: %[[PTEE_TYPE:.*]] = phi i64 [ 32,
// CHECK: %[[PTEE_MODS:.*]] = and i64 %{{.*}}, 1036
// CHECK: %[[PTEE_WITH_MODS:.*]] = or i64 %[[PTEE_TYPE]], %[[PTEE_MODS]]
// CHECK: call void @__tgt_push_mapper_component({{.*}}i64 %[[PTEE_WITH_MODS]],

// The x/y entries are MEMBER_OF the pointee block already. HasAttachPtr
// prevents adding an outer MEMBER_OF directly, but OMPIRBuilder still shifts
// the existing MEMBER_OF field by the current mapper component count.
// CHECK: %[[X_INNER:.*]] = add nuw i64 {{[0-9]+}}, %[[SHIFT]]
// CHECK: %[[X_TYPE:.*]] = phi i64 [ {{.*%.*}}, %{{.*}} ], [ {{.*%.*}}, %{{.*}} ], [ {{.*%.*}}, %{{.*}} ], [ %[[X_INNER]], %{{.*}} ]
// CHECK: %[[X_MODS:.*]] = and i64 %{{.*}}, 1036
// CHECK: %[[X_WITH_MODS:.*]] = or i64 %[[X_TYPE]], %[[X_MODS]]
// CHECK: call void @__tgt_push_mapper_component({{.*}}i64 %[[X_WITH_MODS]],
// CHECK: %[[Y_INNER:.*]] = add nuw i64 {{[0-9]+}}, %[[SHIFT]]
// CHECK: %[[Y_TYPE:.*]] = phi i64 [ {{.*%.*}}, %{{.*}} ], [ {{.*%.*}}, %{{.*}} ], [ {{.*%.*}}, %{{.*}} ], [ %[[Y_INNER]], %{{.*}} ]
// CHECK: %[[Y_MODS:.*]] = and i64 %{{.*}}, 1036
// CHECK: %[[Y_WITH_MODS:.*]] = or i64 %[[Y_TYPE]], %[[Y_MODS]]
// CHECK: call void @__tgt_push_mapper_component({{.*}}i64 %[[Y_WITH_MODS]],

// ATTACH entries are forced to keep their unmodified ATTACH map type.
// CHECK: %[[ATTACH_TYPE:.*]] = phi i64 [ 16384,
// CHECK: %[[ATTACH_MODS:.*]] = and i64 %{{.*}}, 1036
// CHECK: %[[ATTACH_WITH_MODS:.*]] = or i64 %[[ATTACH_TYPE]], %[[ATTACH_MODS]]
// CHECK: call void @__tgt_push_mapper_component({{.*}}i64 %[[ATTACH_TYPE]],
