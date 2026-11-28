// RUN: mlir-opt %s -split-input-file -verify-diagnostics

func.func @unsupported_cache_level(%addr: !llvm.ptr<1>) {
  // expected-error @below {{attribute 'cacheLevel' failed to satisfy constraint: NVVM Prefetch Cache Level whose value is one of {L1}}}
  nvvm.prefetch.valid_addr level = L2, %addr : !llvm.ptr<1>
  return
}

// -----

func.func @non_global_pointer(%addr: !llvm.ptr<3>) {
  // expected-error @below {{operand #0 must be LLVM pointer in address space 0 or LLVM pointer in address space 1}}
  nvvm.prefetch.valid_addr level = L1, %addr : !llvm.ptr<3>
  return
}
