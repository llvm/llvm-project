// RUN: mlir-opt %s --verify-roundtrip

/// Check op assembly.
func.func @ptr_add_type_offset(%ptr: !ptr.ptr<#ptr.generic_space>) -> !ptr.ptr<#ptr.generic_space> {
  %off = ptr.type_offset f32 : index
  %res = ptr.ptr_add %ptr, %off : !ptr.ptr<#ptr.generic_space>, index
  %res0 = ptr.ptr_add none %ptr, %off : !ptr.ptr<#ptr.generic_space>, index
  %res1 = ptr.ptr_add nusw %ptr, %off : !ptr.ptr<#ptr.generic_space>, index
  %res2 = ptr.ptr_add nuw %ptr, %off : !ptr.ptr<#ptr.generic_space>, index
  %res3 = ptr.ptr_add inbounds %ptr, %off : !ptr.ptr<#ptr.generic_space>, index
  return %res : !ptr.ptr<#ptr.generic_space>
}

/// Check cast ops assembly.
func.func @cast_ops(%mr: memref<f32, #ptr.generic_space>) -> memref<f32, #ptr.generic_space> {
  %ptr = ptr.to_ptr %mr : memref<f32, #ptr.generic_space> -> !ptr.ptr<#ptr.generic_space>
  %mda = ptr.get_metadata %mr : memref<f32, #ptr.generic_space>
  %res = ptr.from_ptr %ptr metadata %mda : !ptr.ptr<#ptr.generic_space> -> memref<f32, #ptr.generic_space>
  %mr0 = ptr.from_ptr %ptr : !ptr.ptr<#ptr.generic_space> -> memref<f32, #ptr.generic_space>
  return %res : memref<f32, #ptr.generic_space>
}

/// Check load ops assembly.
func.func @load_ops(%arg0: !ptr.ptr<#ptr.generic_space>) -> (f32, f32, f32, f32, f32, i64, i32) {
  %0 = ptr.load %arg0 : !ptr.ptr<#ptr.generic_space> -> f32
  %1 = ptr.load volatile %arg0 : !ptr.ptr<#ptr.generic_space> -> f32
  %2 = ptr.load %arg0 nontemporal : !ptr.ptr<#ptr.generic_space> -> f32
  %3 = ptr.load %arg0 invariant : !ptr.ptr<#ptr.generic_space> -> f32
  %4 = ptr.load %arg0 invariant_group : !ptr.ptr<#ptr.generic_space> -> f32
  %5 = ptr.load %arg0 atomic monotonic alignment = 8 : !ptr.ptr<#ptr.generic_space> -> i64
  %6 = ptr.load volatile %arg0 atomic syncscope("workgroup") acquire nontemporal alignment = 4 : !ptr.ptr<#ptr.generic_space> -> i32
  return %0, %1, %2, %3, %4, %5, %6 : f32, f32, f32, f32, f32, i64, i32
}

/// Check store ops assembly.
func.func @store_ops(%arg0: !ptr.ptr<#ptr.generic_space>, %arg1: f32, %arg2: i64, %arg3: i32) {
  ptr.store %arg1, %arg0 : f32, !ptr.ptr<#ptr.generic_space>
  ptr.store volatile %arg1, %arg0 : f32, !ptr.ptr<#ptr.generic_space>
  ptr.store %arg1, %arg0 nontemporal : f32, !ptr.ptr<#ptr.generic_space>
  ptr.store %arg1, %arg0 invariant_group : f32, !ptr.ptr<#ptr.generic_space>
  ptr.store %arg2, %arg0 atomic monotonic alignment = 8 : i64, !ptr.ptr<#ptr.generic_space>
  ptr.store volatile %arg3, %arg0 atomic syncscope("workgroup") release nontemporal alignment = 4 : i32, !ptr.ptr<#ptr.generic_space>
  return
}
