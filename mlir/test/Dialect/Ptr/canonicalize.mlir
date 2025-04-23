// RUN: mlir-opt --canonicalize %s | FileCheck %s

/// Check `ptr_add` canonicalizer patterns.

// CHECK-LABEL: @zero_offset
// CHECK-SAME: (%[[PTR_0:.*]]: !ptr.ptr<#ptr.generic_space>)
func.func @zero_offset(%ptr: !ptr.ptr<#ptr.generic_space>) -> !ptr.ptr<#ptr.generic_space> {
  // CHECK-NOT: index.constant
  // CHECK-NOT: ptr.ptr_add
  // CHECK: return %[[PTR_0]] : !ptr.ptr<#ptr.generic_space>
  // CHECK: }
  %off = index.constant 0
  %res0 = ptr.ptr_add %ptr, %off : !ptr.ptr<#ptr.generic_space>, index
  return %res0 : !ptr.ptr<#ptr.generic_space>
}
