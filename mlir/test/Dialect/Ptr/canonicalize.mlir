// RUN: mlir-opt --canonicalize %s | FileCheck %s

/// Check `ptradd` and `type_offset` canonicalizer patterns.

// CHECK-LABEL: @ops0
func.func @ops0(%ptr: !ptr.ptr<#ptr.int_space<3>>, %c: i1) -> !ptr.ptr<#ptr.int_space<3>> {
  // CHECK: (%[[PTR_0:.*]]: !ptr.ptr<#ptr.int_space<3>>,
  // CHECK: %[[F32_OFF:.*]] = ptr.type_offset f32 : index
  // CHECK: %[[PTR_1:.*]] = ptr.ptradd %[[PTR_0]], %[[F32_OFF]] : <#ptr.int_space<3>>, index
  // CHECK: %[[PTR_2:.*]] = ptr.ptradd %[[PTR_1]], %[[F32_OFF]] : <#ptr.int_space<3>>, index
  // CHECK: %[[PTR_3:.*]] = scf.if %{{.*}} -> (!ptr.ptr<#ptr.int_space<3>>) {
  // CHECK: %[[PTR_4:.*]] = ptr.ptradd %[[PTR_2]], %[[F32_OFF]] : <#ptr.int_space<3>>, index
  // CHECK: scf.yield %[[PTR_4]] : !ptr.ptr<#ptr.int_space<3>>
  // CHECK: } else {
  // CHECK: scf.yield %[[PTR_0]] : !ptr.ptr<#ptr.int_space<3>>
  // CHECK: }
  // CHECK: return %[[PTR_3]] : !ptr.ptr<#ptr.int_space<3>>
  // CHECK: }
  %off0 = ptr.type_offset f32 : index
  %res0 = ptr.ptradd %ptr, %off0 : !ptr.ptr<#ptr.int_space<3>>, index
  %off1 = ptr.type_offset f32 : index
  %res1 = ptr.ptradd %res0, %off1 : !ptr.ptr<#ptr.int_space<3>>, index
  %res3 = scf.if %c -> !ptr.ptr<#ptr.int_space<3>> {
    %off2 = ptr.type_offset f32 : index
    %res2 = ptr.ptradd %res1, %off2 : !ptr.ptr<#ptr.int_space<3>>, index
    scf.yield %res2 : !ptr.ptr<#ptr.int_space<3>>
  } else {
    scf.yield %ptr : !ptr.ptr<#ptr.int_space<3>>
  }
  %off3 = index.constant 0
  %res4 = ptr.ptradd %res3, %off3 : !ptr.ptr<#ptr.int_space<3>>, index
  return %res4 : !ptr.ptr<#ptr.int_space<3>>
}
