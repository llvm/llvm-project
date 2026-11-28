// RUN: mlir-opt --omp-offload-privatization-prepare --split-input-file %s | FileCheck %s

// The descriptor's base address is shared by the data and attachment maps, as
// well as ordinary host accesses. Only the target's maps should use the heap
// copy, and the new address must dominate both maps. Capture the descriptor
// after the intervening host store, immediately before the target.
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<64> : vector<4xi64>>} {
  llvm.func @consume(!llvm.ptr)
  omp.private {type = firstprivate} @firstprivate : !llvm.struct<(ptr)> copy {
  ^bb0(%src: !llvm.ptr, %dst: !llvm.ptr):
    %value = llvm.load %src : !llvm.ptr -> !llvm.struct<(ptr)>
    llvm.store %value, %dst : !llvm.struct<(ptr)>, !llvm.ptr
    omp.yield(%dst : !llvm.ptr)
  } dealloc {
  ^bb0(%arg0: !llvm.ptr):
    omp.yield
  }

  // CHECK-LABEL: llvm.func @shared_address(
  // CHECK-SAME: %[[SRC:.*]]: !llvm.ptr, %[[DEP:.*]]: !llvm.ptr, %[[VALUE:.*]]: !llvm.ptr)
  // CHECK: %[[HOST_ADDR:.*]] = llvm.getelementptr %[[SRC]][0, 0]
  // CHECK: %[[BEFORE:.*]] = llvm.load %[[HOST_ADDR]]
  // CHECK: llvm.call @consume(%[[BEFORE]])
  // CHECK: llvm.store %[[VALUE]], %[[HOST_ADDR]]
  // CHECK: %[[HEAP:.*]] = llvm.call @malloc
  // CHECK: %[[COPY:.*]] = llvm.call @firstprivate_copy(%[[SRC]], %[[HEAP]])
  // CHECK: %[[HEAP_ADDR:.*]] = llvm.getelementptr %[[HEAP]][0, 0]
  // CHECK-NEXT: %[[DATA:.*]] = omp.map.info var_ptr(%[[HEAP]] : {{.*}}) map_clauses(to) capture(ByRef) var_ptr_ptr(%[[HEAP_ADDR]] :
  // CHECK-NEXT: %[[DESC:.*]] = omp.map.info var_ptr(%[[HEAP]] : {{.*}}) map_clauses(always, to) capture(ByRef) members(%[[DATA]] :
  // CHECK-NEXT: %[[ATTACH:.*]] = omp.map.info var_ptr(%[[HEAP]] : {{.*}}) map_clauses(attach, ref_ptr, ref_ptee) capture(ByRef) var_ptr_ptr(%[[HEAP_ADDR]] :
  // CHECK-NEXT: omp.target kernel_type(generic) depend(taskdependinout -> %[[DEP]] : !llvm.ptr, taskdependout -> %[[HEAP]] : !llvm.ptr) nowait
  // CHECK-SAME: map_entries(%[[DESC]] -> {{.*}}, %[[DATA]] -> {{.*}}, %[[ATTACH]] ->
  // CHECK-SAME: private(@firstprivate %[[HEAP]] ->
  // CHECK: omp.task depend(taskdependin -> %[[HEAP]] : !llvm.ptr)
  // CHECK: llvm.call @firstprivate_dealloc(%[[COPY]])
  // CHECK: llvm.call @free(%[[HEAP]])
  // CHECK: %[[AFTER:.*]] = llvm.load %[[HOST_ADDR]]
  // CHECK: llvm.call @consume(%[[AFTER]])
  llvm.func @shared_address(%src: !llvm.ptr, %dep: !llvm.ptr, %value: !llvm.ptr) {
    %addr = llvm.getelementptr %src[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr)>
    %before = llvm.load %addr : !llvm.ptr -> !llvm.ptr
    llvm.call @consume(%before) : (!llvm.ptr) -> ()
    llvm.store %value, %addr : !llvm.ptr, !llvm.ptr
    %data = omp.map.info var_ptr(%src : !llvm.ptr, !llvm.struct<(ptr)>) map_clauses(to) capture(ByRef) var_ptr_ptr(%addr : !llvm.ptr, i32) -> !llvm.ptr
    %desc = omp.map.info var_ptr(%src : !llvm.ptr, !llvm.struct<(ptr)>) map_clauses(always, to) capture(ByRef) members(%data : [0] : !llvm.ptr) -> !llvm.ptr
    %attach = omp.map.info var_ptr(%src : !llvm.ptr, !llvm.struct<(ptr)>) map_clauses(attach, ref_ptr, ref_ptee) capture(ByRef) var_ptr_ptr(%addr : !llvm.ptr, i32) -> !llvm.ptr
    omp.target kernel_type(generic) depend(taskdependinout -> %dep : !llvm.ptr) nowait map_entries(%desc -> %map0, %data -> %map1, %attach -> %map2 : !llvm.ptr, !llvm.ptr, !llvm.ptr) private(@firstprivate %src -> %priv [map_idx=0] : !llvm.ptr) {
      omp.terminator
    }
    %after = llvm.load %addr : !llvm.ptr -> !llvm.ptr
    llvm.call @consume(%after) : (!llvm.ptr) -> ()
    llvm.return
  }
}

// -----

// An attachment with a separate address calculation also needs to be remapped,
// even though leaving it alone would not cause a dominance error.
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<64> : vector<4xi64>>} {
  omp.private {type = firstprivate} @firstprivate : !llvm.struct<(ptr)> copy {
  ^bb0(%src: !llvm.ptr, %dst: !llvm.ptr):
    %value = llvm.load %src : !llvm.ptr -> !llvm.struct<(ptr)>
    llvm.store %value, %dst : !llvm.struct<(ptr)>, !llvm.ptr
    omp.yield(%dst : !llvm.ptr)
  } dealloc {
  ^bb0(%arg0: !llvm.ptr):
    omp.yield
  }

  // CHECK-LABEL: llvm.func @separate_addresses(
  // CHECK-SAME: %[[SRC:.*]]: !llvm.ptr)
  // CHECK: %[[HEAP:.*]] = llvm.call @malloc
  // CHECK: llvm.call @firstprivate_copy(%[[SRC]], %[[HEAP]])
  // CHECK: %[[DATA_ADDR:.*]] = llvm.getelementptr %[[HEAP]][0, 0]
  // CHECK-NEXT: %[[DATA:.*]] = omp.map.info var_ptr(%[[HEAP]] : {{.*}}) map_clauses(to) capture(ByRef) var_ptr_ptr(%[[DATA_ADDR]] :
  // CHECK-NEXT: %[[DESC:.*]] = omp.map.info var_ptr(%[[HEAP]] : {{.*}}) map_clauses(always, to) capture(ByRef) members(%[[DATA]] :
  // CHECK-NEXT: %[[ATTACH_ADDR:.*]] = llvm.getelementptr %[[HEAP]][0, 0]
  // CHECK-NEXT: %[[ATTACH:.*]] = omp.map.info var_ptr(%[[HEAP]] : {{.*}}) map_clauses(attach, ref_ptr, ref_ptee) capture(ByRef) var_ptr_ptr(%[[ATTACH_ADDR]] :
  // CHECK-NEXT: omp.target
  // CHECK-SAME: map_entries(%[[DESC]] -> {{.*}}, %[[ATTACH]] -> {{.*}}, %[[DATA]] ->
  llvm.func @separate_addresses(%src: !llvm.ptr) {
    %addr = llvm.getelementptr %src[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr)>
    %data = omp.map.info var_ptr(%src : !llvm.ptr, !llvm.struct<(ptr)>) map_clauses(to) capture(ByRef) var_ptr_ptr(%addr : !llvm.ptr, i32) -> !llvm.ptr
    %desc = omp.map.info var_ptr(%src : !llvm.ptr, !llvm.struct<(ptr)>) map_clauses(always, to) capture(ByRef) members(%data : [0] : !llvm.ptr) -> !llvm.ptr
    %attach_addr = llvm.getelementptr %src[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr)>
    %attach = omp.map.info var_ptr(%src : !llvm.ptr, !llvm.struct<(ptr)>) map_clauses(attach, ref_ptr, ref_ptee) capture(ByRef) var_ptr_ptr(%attach_addr : !llvm.ptr, i32) -> !llvm.ptr
    omp.target kernel_type(generic) nowait map_entries(%desc -> %map0, %attach -> %map1, %data -> %map2 : !llvm.ptr, !llvm.ptr, !llvm.ptr) private(@firstprivate %src -> %priv [map_idx=0] : !llvm.ptr) {
      omp.terminator
    }
    llvm.return
  }
}

// -----

// Maps shared with another target must retain their original descriptor for
// that target. Remapping the deferred target must not replace all their uses.
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<64> : vector<4xi64>>} {
  omp.private {type = firstprivate} @firstprivate : !llvm.struct<(ptr)> copy {
  ^bb0(%src: !llvm.ptr, %dst: !llvm.ptr):
    %value = llvm.load %src : !llvm.ptr -> !llvm.struct<(ptr)>
    llvm.store %value, %dst : !llvm.struct<(ptr)>, !llvm.ptr
    omp.yield(%dst : !llvm.ptr)
  } dealloc {
  ^bb0(%arg0: !llvm.ptr):
    omp.yield
  }

  // CHECK-LABEL: llvm.func @shared_maps(
  // CHECK-SAME: %[[SRC:.*]]: !llvm.ptr)
  // CHECK: %[[HOST_ADDR:.*]] = llvm.getelementptr %[[SRC]][0, 0]
  // CHECK: %[[HOST_DATA:.*]] = omp.map.info var_ptr(%[[SRC]] : {{.*}}) map_clauses(to) capture(ByRef) var_ptr_ptr(%[[HOST_ADDR]] :
  // CHECK: %[[HOST_DESC:.*]] = omp.map.info var_ptr(%[[SRC]] : {{.*}}) map_clauses(always, to) capture(ByRef) members(%[[HOST_DATA]] :
  // CHECK: %[[HOST_ATTACH:.*]] = omp.map.info var_ptr(%[[SRC]] : {{.*}}) map_clauses(attach, ref_ptr, ref_ptee) capture(ByRef) var_ptr_ptr(%[[HOST_ADDR]] :
  // CHECK: %[[HEAP:.*]] = llvm.call @malloc
  // CHECK: %[[HEAP_ADDR:.*]] = llvm.getelementptr %[[HEAP]][0, 0]
  // CHECK: %[[DATA:.*]] = omp.map.info var_ptr(%[[HEAP]] : {{.*}}) map_clauses(to) capture(ByRef) var_ptr_ptr(%[[HEAP_ADDR]] :
  // CHECK: %[[DESC:.*]] = omp.map.info var_ptr(%[[HEAP]] : {{.*}}) map_clauses(always, to) capture(ByRef) members(%[[DATA]] :
  // CHECK: %[[ATTACH:.*]] = omp.map.info var_ptr(%[[HEAP]] : {{.*}}) map_clauses(attach, ref_ptr, ref_ptee) capture(ByRef) var_ptr_ptr(%[[HEAP_ADDR]] :
  // CHECK: omp.target {{.*}} nowait map_entries(%[[DESC]] -> {{.*}}, %[[DATA]] -> {{.*}}, %[[ATTACH]] ->
  // CHECK: omp.task
  // CHECK: omp.target kernel_type(generic) map_entries(%[[HOST_DESC]] -> {{.*}}, %[[HOST_DATA]] -> {{.*}}, %[[HOST_ATTACH]] ->
  llvm.func @shared_maps(%src: !llvm.ptr) {
    %addr = llvm.getelementptr %src[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr)>
    %data = omp.map.info var_ptr(%src : !llvm.ptr, !llvm.struct<(ptr)>) map_clauses(to) capture(ByRef) var_ptr_ptr(%addr : !llvm.ptr, i32) -> !llvm.ptr
    %desc = omp.map.info var_ptr(%src : !llvm.ptr, !llvm.struct<(ptr)>) map_clauses(always, to) capture(ByRef) members(%data : [0] : !llvm.ptr) -> !llvm.ptr
    %attach = omp.map.info var_ptr(%src : !llvm.ptr, !llvm.struct<(ptr)>) map_clauses(attach, ref_ptr, ref_ptee) capture(ByRef) var_ptr_ptr(%addr : !llvm.ptr, i32) -> !llvm.ptr
    omp.target kernel_type(generic) nowait map_entries(%desc -> %map0, %data -> %map1, %attach -> %map2 : !llvm.ptr, !llvm.ptr, !llvm.ptr) private(@firstprivate %src -> %priv [map_idx=0] : !llvm.ptr) {
      omp.terminator
    }
    omp.target kernel_type(generic) map_entries(%desc -> %map0, %data -> %map1, %attach -> %map2 : !llvm.ptr, !llvm.ptr, !llvm.ptr) {
      omp.terminator
    }
    llvm.return
  }
}

// -----

// Remap every operation in the address chain. An independent index produced
// by a call must retain its original evaluation, as must host address users.
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<64> : vector<4xi64>>} {
  llvm.func @next_index() -> i64
  llvm.func @consume(!llvm.ptr)
  omp.private {type = firstprivate} @firstprivate : !llvm.struct<(ptr)> copy {
  ^bb0(%src: !llvm.ptr, %dst: !llvm.ptr):
    %value = llvm.load %src : !llvm.ptr -> !llvm.struct<(ptr)>
    llvm.store %value, %dst : !llvm.struct<(ptr)>, !llvm.ptr
    omp.yield(%dst : !llvm.ptr)
  } dealloc {
  ^bb0(%arg0: !llvm.ptr):
    omp.yield
  }

  // CHECK-LABEL: llvm.func @address_chain(
  // CHECK-SAME: %[[SRC:.*]]: !llvm.ptr)
  // CHECK: %[[INDEX:.*]] = llvm.call @next_index()
  // CHECK: %[[HOST_SLOT:.*]] = llvm.getelementptr %[[SRC]][0, 0]
  // CHECK: %[[HOST_ADDR:.*]] = llvm.getelementptr %[[HOST_SLOT]][%[[INDEX]]]
  // CHECK: llvm.call @consume(%[[HOST_ADDR]])
  // CHECK: %[[HEAP:.*]] = llvm.call @malloc
  // CHECK: llvm.call @firstprivate_copy(%[[SRC]], %[[HEAP]])
  // CHECK-NEXT: %[[HEAP_SLOT:.*]] = llvm.getelementptr %[[HEAP]][0, 0]
  // CHECK-NEXT: %[[HEAP_ADDR:.*]] = llvm.getelementptr %[[HEAP_SLOT]][%[[INDEX]]]
  // CHECK-NEXT: %[[DATA:.*]] = omp.map.info var_ptr(%[[HEAP]] : {{.*}}) map_clauses(to) capture(ByRef) var_ptr_ptr(%[[HEAP_ADDR]] :
  // CHECK-NEXT: %[[DESC:.*]] = omp.map.info var_ptr(%[[HEAP]] : {{.*}}) map_clauses(always, to) capture(ByRef) members(%[[DATA]] :
  // CHECK-NEXT: %[[ATTACH:.*]] = omp.map.info var_ptr(%[[HEAP]] : {{.*}}) map_clauses(attach, ref_ptr, ref_ptee) capture(ByRef) var_ptr_ptr(%[[HEAP_ADDR]] :
  // CHECK-NEXT: omp.target
  // CHECK-SAME: map_entries(%[[DESC]] -> {{.*}}, %[[DATA]] -> {{.*}}, %[[ATTACH]] ->
  // CHECK: llvm.call @consume(%[[HOST_ADDR]])
  // CHECK: llvm.return
  llvm.func @address_chain(%src: !llvm.ptr) {
    %index = llvm.call @next_index() : () -> i64
    %slot = llvm.getelementptr %src[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr)>
    %addr = llvm.getelementptr %slot[%index] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    llvm.call @consume(%addr) : (!llvm.ptr) -> ()
    %data = omp.map.info var_ptr(%src : !llvm.ptr, !llvm.struct<(ptr)>) map_clauses(to) capture(ByRef) var_ptr_ptr(%addr : !llvm.ptr, i32) -> !llvm.ptr
    %desc = omp.map.info var_ptr(%src : !llvm.ptr, !llvm.struct<(ptr)>) map_clauses(always, to) capture(ByRef) members(%data : [0] : !llvm.ptr) -> !llvm.ptr
    %attach = omp.map.info var_ptr(%src : !llvm.ptr, !llvm.struct<(ptr)>) map_clauses(attach, ref_ptr, ref_ptee) capture(ByRef) var_ptr_ptr(%addr : !llvm.ptr, i32) -> !llvm.ptr
    omp.target kernel_type(generic) nowait map_entries(%desc -> %map0, %data -> %map1, %attach -> %map2 : !llvm.ptr, !llvm.ptr, !llvm.ptr) private(@firstprivate %src -> %priv [map_idx=0] : !llvm.ptr) {
      omp.terminator
    }
    llvm.call @consume(%addr) : (!llvm.ptr) -> ()
    llvm.return
  }
}
