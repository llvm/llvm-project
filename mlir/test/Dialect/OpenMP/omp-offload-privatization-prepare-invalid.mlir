// RUN: mlir-opt --omp-offload-privatization-prepare --split-input-file --verify-diagnostics %s

// A descriptor-taking call cannot be re-executed with the heap descriptor.
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<64> : vector<4xi64>>} {
  llvm.func @get_address(!llvm.ptr) -> !llvm.ptr
  llvm.func @consume(!llvm.ptr)
  omp.private {type = firstprivate} @firstprivate : !llvm.struct<(ptr)> copy {
  ^bb0(%src: !llvm.ptr, %dst: !llvm.ptr):
    %value = llvm.load %src : !llvm.ptr -> !llvm.struct<(ptr)>
    llvm.store %value, %dst : !llvm.struct<(ptr)>, !llvm.ptr
    omp.yield(%dst : !llvm.ptr)
  }
  llvm.func @effectful_address(%src: !llvm.ptr) {
    // expected-error @+1 {{cannot safely remap address computation for a deferred target private variable}}
    %addr = llvm.call @get_address(%src) : (!llvm.ptr) -> !llvm.ptr
    %map = omp.map.info var_ptr(%src : !llvm.ptr, !llvm.struct<(ptr)>) map_clauses(to) capture(ByRef) var_ptr_ptr(%addr : !llvm.ptr, i32) -> !llvm.ptr
    omp.target kernel_type(generic) nowait map_entries(%map -> %map0 : !llvm.ptr) private(@firstprivate %src -> %priv [map_idx=0] : !llvm.ptr) {
      omp.terminator
    }
    llvm.call @consume(%addr) : (!llvm.ptr) -> ()
    llvm.return
  }
}

// -----

// Volatile loads must not be duplicated either.
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<64> : vector<4xi64>>} {
  llvm.func @consume(!llvm.ptr)
  omp.private {type = firstprivate} @firstprivate : !llvm.struct<(ptr)> copy {
  ^bb0(%src: !llvm.ptr, %dst: !llvm.ptr):
    %value = llvm.load %src : !llvm.ptr -> !llvm.struct<(ptr)>
    llvm.store %value, %dst : !llvm.struct<(ptr)>, !llvm.ptr
    omp.yield(%dst : !llvm.ptr)
  }
  llvm.func @volatile_address(%src: !llvm.ptr) {
    // expected-error @+1 {{cannot safely remap address computation for a deferred target private variable}}
    %addr = llvm.load volatile %src : !llvm.ptr -> !llvm.ptr
    %map = omp.map.info var_ptr(%src : !llvm.ptr, !llvm.struct<(ptr)>) map_clauses(to) capture(ByRef) var_ptr_ptr(%addr : !llvm.ptr, i32) -> !llvm.ptr
    omp.target kernel_type(generic) nowait map_entries(%map -> %map0 : !llvm.ptr) private(@firstprivate %src -> %priv [map_idx=0] : !llvm.ptr) {
      omp.terminator
    }
    llvm.call @consume(%addr) : (!llvm.ptr) -> ()
    llvm.return
  }
}

// -----

// Do not silently retain an address whose relation to the descriptor is unknown.
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<64> : vector<4xi64>>} {
  omp.private {type = firstprivate} @firstprivate : !llvm.struct<(ptr)> copy {
  ^bb0(%src: !llvm.ptr, %dst: !llvm.ptr):
    %value = llvm.load %src : !llvm.ptr -> !llvm.struct<(ptr)>
    llvm.store %value, %dst : !llvm.struct<(ptr)>, !llvm.ptr
    omp.yield(%dst : !llvm.ptr)
  }
  llvm.func @unknown_address(%src: !llvm.ptr, %addr: !llvm.ptr) {
    // expected-error @+1 {{cannot remap private variable address not derived from its descriptor}}
    %map = omp.map.info var_ptr(%src : !llvm.ptr, !llvm.struct<(ptr)>) map_clauses(to) capture(ByRef) var_ptr_ptr(%addr : !llvm.ptr, i32) -> !llvm.ptr
    omp.target kernel_type(generic) nowait map_entries(%map -> %map0 : !llvm.ptr) private(@firstprivate %src -> %priv [map_idx=0] : !llvm.ptr) {
      omp.terminator
    }
    llvm.return
  }
}
