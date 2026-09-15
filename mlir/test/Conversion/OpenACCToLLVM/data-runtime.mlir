// RUN: mlir-opt %s -acc-to-llvm -split-input-file | FileCheck %s

// Data runtime argument emission preserves every dialect map flag.
// CHECK-LABEL: llvm.func @all_map_flags
// CHECK: %[[FLAGS:.*]] = llvm.mlir.constant(8384411 : i64) : i64
// CHECK: llvm.store %[[FLAGS]], %{{.*}} : i64, !llvm.ptr
// CHECK: llvm.call @__tgt_acc_data_enter
func.func @all_map_flags(%arg0: !llvm.ptr) {
  %size = arith.constant 4 : i64
  %map = acc.map_info varPtr(%arg0 : !llvm.ptr) varType(i32)
      size(%size : i64) elementSize(4)
      descKind(none)
      mapFlags(to, from, delete, ptr_and_obj, private, literal, implicit,
               devptr, managed_devptr, no_create, gang_private,
               worker_private, vector_private, init_zero, device_resident,
               if_present, present, descriptor, reduction) -> !llvm.ptr
  acc.enter_data dataOperands(%map : !llvm.ptr)
  return
}

// -----

// A normal delete decrements the dynamic reference count and therefore does
// not force unmapping. Finalize adds DELETE while retaining FROM for copyout.
// CHECK-LABEL: llvm.func @delete_and_finalize
// CHECK: %[[SIZE0:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT: %[[NONE:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK: llvm.store %[[NONE]], %{{.*}} : i64, !llvm.ptr
// CHECK: llvm.call @__tgt_acc_data_exit
// CHECK: %[[SIZE1:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT: %[[FROM_DELETE:.*]] = llvm.mlir.constant(10 : i64) : i64
// CHECK: llvm.store %[[FROM_DELETE]], %{{.*}} : i64, !llvm.ptr
// CHECK: llvm.call @__tgt_acc_data_exit
func.func @delete_and_finalize(%arg0: !llvm.ptr) {
  %delete = acc.getdeviceptr varPtr(%arg0 : !llvm.ptr) varType(i32)
      dataClause(acc_delete) structured(false) -> !llvm.ptr
  acc.exit_data dataOperands(%delete : !llvm.ptr)
  acc.delete accPtr(%delete : !llvm.ptr) dataClause(acc_delete)
      structured(false)

  %copyout = acc.getdeviceptr varPtr(%arg0 : !llvm.ptr) varType(i32)
      dataClause(acc_copyout) structured(false) -> !llvm.ptr
  acc.exit_data dataOperands(%copyout : !llvm.ptr) finalize
  acc.copyout accPtr(%copyout : !llvm.ptr) to varPtr(%arg0 : !llvm.ptr)
      varType(i32) dataClause(acc_copyout) structured(false)
  return
}

// -----

// Clause operations derive the same map flags as prepared map_info operations.
// A zero-initialized create carries INIT_ZERO, and reduction carries TO, FROM,
// IMPLICIT, and REDUCTION through both calls of a structured data region.
// CHECK-LABEL: llvm.func @clause_map_flags
// CHECK-DAG: %[[ZERO:.*]] = llvm.mlir.constant(131072 : i64) : i64
// CHECK-DAG: llvm.store %[[ZERO]], %{{.*}} : i64, !llvm.ptr
// CHECK-DAG: %[[REDUCTION:.*]] = llvm.mlir.constant(4194819 : i64) : i64
// CHECK-DAG: llvm.store %[[REDUCTION]], %{{.*}} : i64, !llvm.ptr
// CHECK: llvm.call @__tgt_acc_data_begin
// CHECK: %[[REDUCTION_END:.*]] = llvm.mlir.constant(4194819 : i64) : i64
// CHECK: llvm.store %[[REDUCTION_END]], %{{.*}} : i64, !llvm.ptr
// CHECK: llvm.call @__tgt_acc_data_end
func.func @clause_map_flags(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
  %zero = acc.create varPtr(%arg0 : !llvm.ptr) varType(i32)
      dataClause(acc_create_zero) -> !llvm.ptr
  %reduction = acc.copyin varPtr(%arg1 : !llvm.ptr) varType(i32)
      dataClause(acc_reduction) implicit(true) -> !llvm.ptr
  acc.data dataOperands(%zero, %reduction : !llvm.ptr, !llvm.ptr) {
    acc.terminator
  }
  acc.delete accPtr(%zero : !llvm.ptr) dataClause(acc_create_zero)
  acc.copyout accPtr(%reduction : !llvm.ptr) to varPtr(%arg1 : !llvm.ptr)
      varType(i32) dataClause(acc_reduction) implicit(true)
  return
}

// -----

// ifPresent is a property of the update construct and applies to each of its
// mappings in addition to the direction stated by its clause.
// CHECK-LABEL: llvm.func @update_if_present
// CHECK: %[[TO_PRESENT:.*]] = llvm.mlir.constant(524289 : i64) : i64
// CHECK: llvm.store %[[TO_PRESENT]], %{{.*}} : i64, !llvm.ptr
// CHECK: llvm.call @__tgt_acc_data_update
// CHECK: %[[FROM_PRESENT:.*]] = llvm.mlir.constant(524290 : i64) : i64
// CHECK: llvm.store %[[FROM_PRESENT]], %{{.*}} : i64, !llvm.ptr
// CHECK: llvm.call @__tgt_acc_data_update
func.func @update_if_present(%arg0: !llvm.ptr) {
  %to = acc.update_device varPtr(%arg0 : !llvm.ptr) varType(i32)
      structured(false) -> !llvm.ptr
  acc.update dataOperands(%to : !llvm.ptr) ifPresent
  %from = acc.getdeviceptr varPtr(%arg0 : !llvm.ptr) varType(i32)
      dataClause(acc_update_host) structured(false) -> !llvm.ptr
  acc.update dataOperands(%from : !llvm.ptr) ifPresent
  acc.update_host accPtr(%from : !llvm.ptr) to varPtr(%arg0 : !llvm.ptr)
      varType(i32) structured(false)
  return
}

// -----

// Bounds are packed in source order. sourceExtent describes the whole source
// object, and an element stride is converted to bytes.
// CHECK-LABEL: llvm.func @bounded_map
// CHECK-DAG: %[[LB:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-DAG: %[[UB:.*]] = llvm.mlir.constant(6 : i64) : i64
// CHECK-DAG: %[[SOURCE_EXTENT:.*]] = llvm.mlir.constant(20 : i64) : i64
// CHECK-DAG: %[[STRIDE:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-DAG: %[[ELEMENT_SIZE:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-DAG: %[[BYTE_STRIDE:.*]] = llvm.mul %[[STRIDE]], %[[ELEMENT_SIZE]] : i64
// CHECK-DAG: llvm.store %[[LB]], %{{.*}} : i64, !llvm.ptr
// CHECK-DAG: llvm.store %[[UB]], %{{.*}} : i64, !llvm.ptr
// CHECK-DAG: llvm.store %[[SOURCE_EXTENT]], %{{.*}} : i64, !llvm.ptr
// CHECK-DAG: llvm.store %[[BYTE_STRIDE]], %{{.*}} : i64, !llvm.ptr
// CHECK-DAG: %[[RANK:.*]] = llvm.mlir.constant(1 : i8) : i8
// CHECK-DAG: llvm.insertvalue %[[RANK]], %{{.*}}[1]
// CHECK-DAG: llvm.call @__tgt_acc_data_begin
func.func @bounded_map(%arg0: !llvm.ptr) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %c20 = arith.constant 20 : index
  %bound = acc.bounds lowerbound(%c2 : index) upperbound(%c6 : index)
      extent(%c5 : index) sourceExtent(%c20 : index)
      stride(%c3 : index) startIdx(%c1 : index)
  %map = acc.map_info varPtr(%arg0 : !llvm.ptr) varType(tensor<?xi32>)
      bounds(%bound) elementSize(4) descKind(openacc)
      mapFlags(to, from) -> !llvm.ptr
  acc.data dataOperands(%map : !llvm.ptr) {
    acc.terminator
  }
  return
}

// -----

// A CFI-backed mapping records the CFI descriptor pointer. With OpenACC
// bounds, the base descriptor version contains both descriptor-kind bits.
// CHECK-LABEL: llvm.func @cfi_descriptor
// CHECK: %[[CFI:.*]] = llvm.mlir.zero : !llvm.struct<(i32, ptr)>
// CHECK: %[[DESC_PTR:.*]] = llvm.insertvalue %arg1, %[[CFI]][1]
// cfi | openacc = 1 | 4096.
// CHECK: %[[VERSION:.*]] = llvm.mlir.constant(4097 : i32) : i32
// CHECK: %[[VERSIONED:.*]] = llvm.insertvalue %[[VERSION]], %[[DESC_PTR]][0]
// CHECK: llvm.insertvalue %[[VERSIONED]], %{{.*}}[0]
// CHECK: llvm.call @__tgt_acc_data_begin
func.func @cfi_descriptor(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c7 = arith.constant 7 : index
  %c8 = arith.constant 8 : index
  %bound = acc.bounds lowerbound(%c0 : index) upperbound(%c7 : index)
      extent(%c8 : index) stride(%c1 : index) startIdx(%c1 : index)
  %map = acc.map_info varPtr(%arg0 : !llvm.ptr) varType(tensor<?xf64>)
      desc(%arg1 : !llvm.ptr) bounds(%bound) elementSize(8)
      descKind(cfi, openacc) mapFlags(to, from, descriptor) -> !llvm.ptr
  acc.data dataOperands(%map : !llvm.ptr) {
    acc.terminator
  }
  return
}

// -----

// varPtrPtr is the attachment address and therefore supplies ArgBasePtrs.
// The descriptor itself is used as the base for PTR_AND_OBJ when there is no
// attachment address.
// A CFI descriptor without bounds keeps the CFI-only version, unlike the
// combined version a mapping with OpenACC bounds states.
// CHECK-LABEL: llvm.func @mapping_bases
// CHECK: %[[CFI:.*]] = llvm.mlir.zero : !llvm.struct<(i32, ptr)>
// CHECK: %[[CFI_PTR:.*]] = llvm.insertvalue %arg2, %[[CFI]][1]
// CHECK: %[[CFI_VERSION:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: llvm.insertvalue %[[CFI_VERSION]], %[[CFI_PTR]][0]
// CHECK: llvm.store %arg1, %{{.*}} : !llvm.ptr, !llvm.ptr
// CHECK: llvm.store %arg4, %{{.*}} : !llvm.ptr, !llvm.ptr
// CHECK: llvm.call @__tgt_acc_data_begin
func.func @mapping_bases(%arg0: !llvm.ptr, %arg1: !llvm.ptr,
    %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr) {
  %size = arith.constant 8 : i64
  %attached = acc.map_info varPtr(%arg0 : !llvm.ptr) varType(i64)
      varPtrPtr(%arg1 : !llvm.ptr) desc(%arg2 : !llvm.ptr)
      size(%size : i64) descKind(cfi)
      mapFlags(to, ptr_and_obj) -> !llvm.ptr
  %descriptor = acc.map_info varPtr(%arg3 : !llvm.ptr) varType(i64)
      desc(%arg4 : !llvm.ptr) size(%size : i64) descKind(cfi)
      mapFlags(to, ptr_and_obj) -> !llvm.ptr
  acc.data dataOperands(%attached, %descriptor : !llvm.ptr, !llvm.ptr) {
    acc.terminator
  }
  return
}

// -----

// A local device-resident object has no host base and is forcibly removed when
// its structured lifetime ends.
// CHECK-LABEL: llvm.func @device_resident_exit
// CHECK: %[[RESIDENT:.*]] = llvm.mlir.constant(262144 : i64) : i64
// CHECK: llvm.call @__tgt_acc_data_begin
// CHECK: %[[LOCAL_DELETE:.*]] = llvm.mlir.constant(262152 : i64) : i64
// CHECK: llvm.store %[[LOCAL_DELETE]], %{{.*}} : i64, !llvm.ptr
// CHECK: llvm.call @__tgt_acc_data_end
func.func @device_resident_exit(%local: !llvm.ptr) {
  %size = arith.constant 4 : i64
  %localMap = acc.map_info varPtr(%local : !llvm.ptr) varType(i32)
      size(%size : i64) elementSize(4) descKind(none)
      mapFlags(device_resident) -> !llvm.ptr
  acc.data dataOperands(%localMap : !llvm.ptr) {
    acc.terminator
  }
  return
}

// -----

// Runtime variable-name globals disambiguate names that sanitize to the same
// symbol while reusing a global for repeated names with the same contents.
// CHECK-DAG: llvm.mlir.global internal constant @acc.var_name.a_b("a-b\00")
// CHECK-DAG: llvm.mlir.global internal constant @acc.var_name.a_b.0("a_b\00")
// CHECK-NOT: @acc.var_name.a_b.1
// CHECK-LABEL: llvm.func @mapping_name_collisions
// CHECK: llvm.call @__tgt_acc_data_enter
func.func @mapping_name_collisions(%arg0: !llvm.ptr, %arg1: !llvm.ptr,
    %arg2: !llvm.ptr) {
  %size = arith.constant 4 : i64
  %first = acc.map_info varPtr(%arg0 : !llvm.ptr) varType(i32)
      size(%size : i64) name("a-b") descKind(none)
      mapFlags(to) -> !llvm.ptr
  %second = acc.map_info varPtr(%arg1 : !llvm.ptr) varType(i32)
      size(%size : i64) name("a_b") descKind(none)
      mapFlags(to) -> !llvm.ptr
  %repeat = acc.map_info varPtr(%arg2 : !llvm.ptr) varType(i32)
      size(%size : i64) name("a-b") descKind(none)
      mapFlags(to) -> !llvm.ptr
  acc.enter_data dataOperands(%first, %second, %repeat
      : !llvm.ptr, !llvm.ptr, !llvm.ptr)
  return
}

// -----

// A structured mapping reports the entry operation at data_begin and its
// exitLoc at data_end.
// CHECK-DAG: llvm.mlir.global internal constant @acc.loc.3.2.{{[0-9]+}}(";begin.mlir;mapping_exit_location;3;2;;\00")
// CHECK-DAG: llvm.mlir.global internal constant @acc.loc.9.4.{{[0-9]+}}(";end.mlir;mapping_exit_location;9;4;;\00")
// CHECK-LABEL: llvm.func @mapping_exit_location
// CHECK: llvm.mlir.addressof @acc.ident.3.2.{{[0-9]+}}
// CHECK: llvm.call @__tgt_acc_data_begin
// CHECK: llvm.mlir.addressof @acc.ident.9.4.{{[0-9]+}}
// CHECK: llvm.call @__tgt_acc_data_end
func.func @mapping_exit_location(%arg0: !llvm.ptr) {
  %size = arith.constant 4 : i64
  %map = acc.map_info varPtr(%arg0 : !llvm.ptr) varType(i32)
      size(%size : i64) exitLoc(loc("end.mlir":9:4)) descKind(none)
      mapFlags(to, from) -> !llvm.ptr loc("begin.mlir":3:2)
  acc.data dataOperands(%map : !llvm.ptr) {
    acc.terminator
  }
  return
}

// -----

// An empty data region is only a lifetime/control-flow scope. It needs no
// runtime mapping call, but its body still executes.
// CHECK-LABEL: llvm.func @empty_data
// CHECK-NOT: __tgt_acc_data_
// CHECK: llvm.return
func.func @empty_data() {
  acc.data {
    acc.terminator
  } defaultAttr(none)
  return
}

// -----

// A memref is converted to a multi-field LLVM aggregate. The lowering emits
// one mapped object per field, so the mapping arrays have that many entries.
// CHECK-LABEL: llvm.func @memref_fields
// CHECK-COUNT-5: llvm.extractvalue
// CHECK: %[[FIVE:.*]] = llvm.mlir.constant(5 : i32) : i32
// CHECK: %[[ELEMENT_SIZE:.*]] = llvm.mlir.constant(8 : i64) : i64
// CHECK: %[[MEMREF_DESC:.*]] = llvm.mlir.zero : !llvm.struct<(i32, i8, i64, ptr)>
// CHECK: %[[RANK:.*]] = llvm.mlir.constant(1 : i8) : i8
// CHECK: %[[WITH_RANK:.*]] = llvm.insertvalue %[[RANK]], %[[MEMREF_DESC]][1]
// CHECK: %[[WITH_SIZE:.*]] = llvm.insertvalue %[[ELEMENT_SIZE]], %[[WITH_RANK]][2]
// CHECK: %[[MEMREF_VERSION:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK: llvm.insertvalue %[[MEMREF_VERSION]], %{{.*}}[0]
// CHECK: llvm.call @__tgt_acc_data_begin
func.func @memref_fields(%arg0: memref<?xf64>) {
  %map = acc.copyin varPtr(%arg0 : memref<?xf64>)
      dataClause(acc_copyin) -> memref<?xf64>
  acc.data dataOperands(%map : memref<?xf64>) {
    acc.terminator
  }
  acc.delete accPtr(%map : memref<?xf64>) dataClause(acc_copyin)
  return
}

// -----

// Every acc.terminator in a multi-block structured region branches to the
// continuation after the data-end call.
// CHECK-LABEL: llvm.func @multiblock_data
// CHECK: llvm.call @__tgt_acc_data_begin
// CHECK: llvm.br ^[[BODY:bb[0-9]+]]
// CHECK: ^[[BODY]]:
// CHECK: llvm.cond_br %arg1, ^[[LEFT:bb[0-9]+]], ^[[RIGHT:bb[0-9]+]]
// CHECK: ^[[LEFT]]:
// CHECK: llvm.br ^[[CONT:bb[0-9]+]]
// CHECK: ^[[RIGHT]]:
// CHECK: llvm.br ^[[CONT]]
// CHECK: ^[[CONT]]:
// CHECK: llvm.call @__tgt_acc_data_end
func.func @multiblock_data(%arg0: !llvm.ptr, %cond: i1) {
  %size = arith.constant 4 : i64
  %map = acc.map_info varPtr(%arg0 : !llvm.ptr) varType(i32)
      size(%size : i64) elementSize(4)
      descKind(none)
      mapFlags(to, from) -> !llvm.ptr
  acc.data dataOperands(%map : !llvm.ptr) {
    cf.cond_br %cond, ^left, ^right
  ^left:
    acc.terminator
  ^right:
    acc.terminator
  }
  return
}

// -----

// An object mapped by value is sized from what is passed to the runtime rather
// than from the stated size of the object it was read from.
// CHECK-LABEL: llvm.func @literal_size
// CHECK: %[[SIZE:.*]] = llvm.mlir.constant(8 : i64) : i64
// CHECK: llvm.store %[[SIZE]], %{{.*}} : i64, !llvm.ptr
// CHECK: llvm.call @__tgt_acc_data_enter
func.func @literal_size(%arg0: !llvm.ptr) {
  %size = arith.constant 4 : i64
  %map = acc.map_info varPtr(%arg0 : !llvm.ptr) varType(i32)
      size(%size : i64) elementSize(4) descKind(none)
      mapFlags(to, literal) -> !llvm.ptr
  acc.enter_data dataOperands(%map : !llvm.ptr)
  return
}

// -----

// A stride already expressed in bytes is copied directly into the descriptor.
// CHECK-LABEL: llvm.func @byte_stride
// CHECK: %[[STRIDE:.*]] = llvm.mlir.constant(16 : i64) : i64
// CHECK-NOT: llvm.mul
// CHECK: llvm.store %[[STRIDE]], %{{.*}} : i64, !llvm.ptr
// CHECK: llvm.call @__tgt_acc_data_enter
func.func @byte_stride(%arg0: !llvm.ptr) {
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %bound = acc.bounds lowerbound(%c0 : index) upperbound(%c3 : index)
      extent(%c4 : index) stride(%c16 : index) startIdx(%c0 : index)
      strideInBytes(true)
  %map = acc.map_info varPtr(%arg0 : !llvm.ptr) varType(tensor<?xi32>)
      bounds(%bound) elementSize(4) descKind(openacc)
      mapFlags(to) -> !llvm.ptr
  acc.enter_data dataOperands(%map : !llvm.ptr)
  return
}

// -----

// Extents retain every full source dimension for a multidimensional section.
// CHECK-LABEL: llvm.func @multidimensional_extents
// CHECK-DAG: %[[E5:.*]] = llvm.mlir.constant(5 : i64) : i64
// CHECK-DAG: %[[E12:.*]] = llvm.mlir.constant(12 : i64) : i64
// CHECK-DAG: %[[E13:.*]] = llvm.mlir.constant(13 : i64) : i64
// CHECK: %[[RANK:.*]] = llvm.mlir.constant(4 : i8) : i8
// CHECK: llvm.store %[[E13]], %{{.*}} : i64, !llvm.ptr
// CHECK: llvm.store %[[E13]], %{{.*}} : i64, !llvm.ptr
// CHECK: llvm.store %[[E12]], %{{.*}} : i64, !llvm.ptr
// CHECK: llvm.store %[[E5]], %{{.*}} : i64, !llvm.ptr
// CHECK: llvm.insertvalue %[[RANK]], %{{.*}}[1]
// CHECK: llvm.call @__tgt_acc_data_enter
func.func @multidimensional_extents(%arg0: !llvm.ptr) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c11 = arith.constant 11 : index
  %c12 = arith.constant 12 : index
  %c13 = arith.constant 13 : index
  %b0 = acc.bounds lowerbound(%c0 : index) upperbound(%c4 : index)
      extent(%c5 : index) sourceExtent(%c13 : index)
      stride(%c1 : index) startIdx(%c0 : index)
  %b1 = acc.bounds lowerbound(%c0 : index) upperbound(%c4 : index)
      extent(%c5 : index) sourceExtent(%c13 : index)
      stride(%c1 : index) startIdx(%c0 : index)
  %b2 = acc.bounds lowerbound(%c0 : index) upperbound(%c11 : index)
      extent(%c12 : index) sourceExtent(%c12 : index)
      stride(%c1 : index) startIdx(%c0 : index)
  %b3 = acc.bounds lowerbound(%c0 : index) upperbound(%c4 : index)
      extent(%c5 : index) sourceExtent(%c5 : index)
      stride(%c1 : index) startIdx(%c0 : index)
  %map = acc.map_info varPtr(%arg0 : !llvm.ptr)
      varType(tensor<?x?x?x?xf32>) bounds(%b0, %b1, %b2, %b3)
      elementSize(4) descKind(openacc) mapFlags(to) -> !llvm.ptr
  acc.enter_data dataOperands(%map : !llvm.ptr)
  return
}

// -----

memref.global @device_global : memref<4xf32> = uninitialized

// A device-resident object reached through a global symbol outlives the region
// that states it, so its mapping is not turned into a deletion the way the
// mapping of a local one is.
// CHECK-LABEL: llvm.func @global_device_resident
// CHECK: %[[ENTER:.*]] = llvm.mlir.constant(262144 : i64) : i64
// CHECK: llvm.store %[[ENTER]], %{{.*}} : i64, !llvm.ptr
// CHECK: llvm.call @__tgt_acc_data_begin
// CHECK: %[[EXIT:.*]] = llvm.mlir.constant(262144 : i64) : i64
// CHECK: llvm.store %[[EXIT]], %{{.*}} : i64, !llvm.ptr
// CHECK-NOT: llvm.mlir.constant(262152 : i64) : i64
// CHECK: llvm.call @__tgt_acc_data_end
func.func @global_device_resident() {
  %global = memref.get_global @device_global : memref<4xf32>
  %map = acc.map_info varPtr(%global : memref<4xf32>) varType(f32)
      elementSize(4) descKind(none)
      mapFlags(device_resident) -> memref<4xf32>
  acc.data dataOperands(%map : memref<4xf32>) {
    acc.terminator
  }
  return
}

// -----

// Named mappings become string globals whose addresses fill ArgNames; a mapping
// that states no name leaves a null slot instead, so no second name global is
// created.
// CHECK: llvm.mlir.global internal constant @acc.var_name.x("x\00")
// CHECK-NOT: @acc.var_name.
// CHECK-LABEL: llvm.func @mapping_names
// CHECK: %[[NAME:.*]] = llvm.mlir.addressof @acc.var_name.x
// CHECK: %[[GEP:.*]] = llvm.getelementptr %[[NAME]]
// CHECK: llvm.store %[[GEP]], %{{.*}} : !llvm.ptr, !llvm.ptr
// CHECK: llvm.call @__tgt_acc_data_enter
func.func @mapping_names(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
  %size = arith.constant 4 : i64
  %named = acc.map_info varPtr(%arg0 : !llvm.ptr) varType(i32)
      size(%size : i64) name("x") descKind(none)
      mapFlags(to) -> !llvm.ptr
  %unnamed = acc.map_info varPtr(%arg1 : !llvm.ptr) varType(i32)
      size(%size : i64) descKind(none)
      mapFlags(to) -> !llvm.ptr
  acc.enter_data dataOperands(%named, %unnamed : !llvm.ptr, !llvm.ptr)
  return
}

// -----

// create allocates without a copy, copyin_readonly copies to the device, and
// attach/detach state the attachment address as PTR_AND_OBJ.
// CHECK-LABEL: llvm.func @create_readonly_detach
// CHECK-DAG: %[[TO:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG: %[[PTR_AND_OBJ:.*]] = llvm.mlir.constant(16 : i64) : i64
// CHECK-DAG: llvm.store %[[TO]], %{{.*}} : i64, !llvm.ptr
// CHECK-DAG: llvm.store %[[PTR_AND_OBJ]], %{{.*}} : i64, !llvm.ptr
// CHECK-NOT: llvm.mlir.constant(3 : i64)
// CHECK: llvm.call @__tgt_acc_data_begin
func.func @create_readonly_detach(%arg0: !llvm.ptr, %arg1: !llvm.ptr,
    %arg2: !llvm.ptr) {
  %create = acc.create varPtr(%arg0 : !llvm.ptr) varType(i32) -> !llvm.ptr
  %readonly = acc.copyin varPtr(%arg1 : !llvm.ptr) varType(i32)
      dataClause(acc_copyin_readonly) -> !llvm.ptr
  %attach = acc.attach varPtr(%arg2 : !llvm.ptr) varType(i32) -> !llvm.ptr
  acc.data dataOperands(%create, %readonly, %attach
      : !llvm.ptr, !llvm.ptr, !llvm.ptr) {
    acc.terminator
  }
  acc.delete accPtr(%create : !llvm.ptr)
  acc.delete accPtr(%readonly : !llvm.ptr) dataClause(acc_copyin_readonly)
  acc.detach accPtr(%attach : !llvm.ptr)
  return
}

// -----

// Bounds on a memref wrap the memref descriptor in the OpenACC overlay, so the
// version carries both kinds.
// CHECK-LABEL: llvm.func @memref_bounded
// CHECK-COUNT-5: llvm.extractvalue
// CHECK: %[[ELEMENT_SIZE:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK: %[[MEMREF_DESC:.*]] = llvm.mlir.zero : !llvm.struct<(i32, i8, i64, ptr)>
// CHECK: %[[RANK:.*]] = llvm.mlir.constant(1 : i8) : i8
// CHECK: %[[WITH_RANK:.*]] = llvm.insertvalue %[[RANK]], %[[MEMREF_DESC]][1]
// CHECK: llvm.insertvalue %[[ELEMENT_SIZE]], %[[WITH_RANK]][2]
// memref | openacc = 2 | 4096.
// CHECK: %[[VERSION:.*]] = llvm.mlir.constant(4098 : i32) : i32
// CHECK: llvm.insertvalue %[[VERSION]], %{{.*}}[0]
// CHECK: llvm.call @__tgt_acc_data_begin
func.func @memref_bounded(%arg0: memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %bound = acc.bounds lowerbound(%c0 : index) upperbound(%c3 : index)
      extent(%c4 : index) stride(%c1 : index) startIdx(%c0 : index)
  %map = acc.copyin varPtr(%arg0 : memref<?xf32>) varType(tensor<?xf32>)
      bounds(%bound) dataClause(acc_copyin) -> memref<?xf32>
  acc.data dataOperands(%map : memref<?xf32>) {
    acc.terminator
  }
  acc.delete accPtr(%map : memref<?xf32>) dataClause(acc_copyin)
  return
}
