// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: define void @tensormap_replace_global_address(ptr addrspace(1) %0, i64 %1) {
llvm.func @tensormap_replace_global_address(%addr : !llvm.ptr<1>, %new_val : i64) {
  // CHECK-NEXT: call void @llvm.nvvm.tensormap.replace.global.address.p1(ptr addrspace(1) %0, i64 %1)
  nvvm.tensormap.replace field = global_address, new_value = %new_val in %addr : !llvm.ptr<1>, i64

  // CHECK-NEXT: ret void
  llvm.return
  // CHECK-NEXT: }
}

// CHECK-LABEL: define void @tensormap_replace_rank(ptr addrspace(1) %0, i32 %1) {
llvm.func @tensormap_replace_rank(%addr : !llvm.ptr<1>, %new_val : i32) {
  // CHECK-NEXT: call void @llvm.nvvm.tensormap.replace.rank.p1(ptr addrspace(1) %0, i32 %1)
  nvvm.tensormap.replace field = rank, new_value = %new_val in %addr : !llvm.ptr<1>, i32

  // CHECK-NEXT: ret void
  llvm.return
  // CHECK-NEXT: }
}

// CHECK-LABEL: define void @tensormap_replace_box_dim(ptr addrspace(1) %0, i32 %1) {
llvm.func @tensormap_replace_box_dim(%addr : !llvm.ptr<1>, %new_val : i32) {
  // CHECK-NEXT: call void @llvm.nvvm.tensormap.replace.box.dim.p1(ptr addrspace(1) %0, i32 1, i32 %1)
  nvvm.tensormap.replace field = box_dim[1], new_value = %new_val in %addr : !llvm.ptr<1>, i32

  // CHECK-NEXT: ret void
  llvm.return
  // CHECK-NEXT: }
}

// CHECK-LABEL: define void @tensormap_replace_global_dim(ptr addrspace(1) %0, i32 %1) {
llvm.func @tensormap_replace_global_dim(%addr : !llvm.ptr<1>, %new_val : i32) {
  // CHECK-NEXT: call void @llvm.nvvm.tensormap.replace.global.dim.p1(ptr addrspace(1) %0, i32 1, i32 %1)
  nvvm.tensormap.replace field = global_dim[1], new_value = %new_val in %addr : !llvm.ptr<1>, i32
  
  // CHECK-NEXT: ret void
  llvm.return
  // CHECK-NEXT: }
}

// CHECK-LABEL: define void @tensormap_replace_global_stride(ptr addrspace(1) %0, i64 %1) {
llvm.func @tensormap_replace_global_stride(%addr : !llvm.ptr<1>, %new_val : i64) {
  // CHECK-NEXT: call void @llvm.nvvm.tensormap.replace.global.stride.p1(ptr addrspace(1) %0, i32 1, i64 %1)
  nvvm.tensormap.replace field = global_stride[1], new_value = %new_val in %addr : !llvm.ptr<1>, i64

  // CHECK-NEXT: ret void
  llvm.return
  // CHECK-NEXT: }
}

// CHECK-LABEL: define void @tensormap_replace_element_stride(ptr addrspace(1) %0, i32 %1) {
llvm.func @tensormap_replace_element_stride(%addr : !llvm.ptr<1>, %new_val : i32) {
  // CHECK-NEXT: call void @llvm.nvvm.tensormap.replace.element.stride.p1(ptr addrspace(1) %0, i32 1, i32 %1)
  nvvm.tensormap.replace field = element_stride[1], new_value = %new_val in %addr : !llvm.ptr<1>, i32

  // CHECK-NEXT: ret void
  llvm.return
  // CHECK-NEXT: }
}

// CHECK-LABEL: define void @tensormap_replace_elemtype(ptr addrspace(1) %0) {
llvm.func @tensormap_replace_elemtype(%addr : !llvm.ptr<1>) {
  // CHECK-NEXT: call void @llvm.nvvm.tensormap.replace.elemtype.p1(ptr addrspace(1) %0, /* elemtype=u8 */ i32 0)
  nvvm.tensormap.replace field = elemtype, new_value = #nvvm.tensormap_elemtype<u8> in %addr : !llvm.ptr<1>

  // CHECK-NEXT: call void @llvm.nvvm.tensormap.replace.elemtype.p1(ptr addrspace(1) %0, /* elemtype=u16 */ i32 1)
  nvvm.tensormap.replace field = elemtype, new_value = #nvvm.tensormap_elemtype<u16> in %addr : !llvm.ptr<1>

  // CHECK-NEXT: call void @llvm.nvvm.tensormap.replace.elemtype.p1(ptr addrspace(1) %0, /* elemtype=u32 */ i32 2)
  nvvm.tensormap.replace field = elemtype, new_value = #nvvm.tensormap_elemtype<u32> in %addr : !llvm.ptr<1>

  // CHECK-NEXT: call void @llvm.nvvm.tensormap.replace.elemtype.p1(ptr addrspace(1) %0, /* elemtype=s32 */ i32 3)
  nvvm.tensormap.replace field = elemtype, new_value = #nvvm.tensormap_elemtype<s32> in %addr : !llvm.ptr<1>

  // CHECK-NEXT: call void @llvm.nvvm.tensormap.replace.elemtype.p1(ptr addrspace(1) %0, /* elemtype=u64 */ i32 4)
  nvvm.tensormap.replace field = elemtype, new_value = #nvvm.tensormap_elemtype<u64> in %addr : !llvm.ptr<1>

  // CHECK-NEXT: call void @llvm.nvvm.tensormap.replace.elemtype.p1(ptr addrspace(1) %0, /* elemtype=s64 */ i32 5)
  nvvm.tensormap.replace field = elemtype, new_value = #nvvm.tensormap_elemtype<s64> in %addr : !llvm.ptr<1>

  // CHECK-NEXT: call void @llvm.nvvm.tensormap.replace.elemtype.p1(ptr addrspace(1) %0, /* elemtype=f16 */ i32 6)
  nvvm.tensormap.replace field = elemtype, new_value = #nvvm.tensormap_elemtype<f16> in %addr : !llvm.ptr<1>

  // CHECK-NEXT: call void @llvm.nvvm.tensormap.replace.elemtype.p1(ptr addrspace(1) %0, /* elemtype=f32 */ i32 7)
  nvvm.tensormap.replace field = elemtype, new_value = #nvvm.tensormap_elemtype<f32> in %addr : !llvm.ptr<1>

  // CHECK-NEXT: call void @llvm.nvvm.tensormap.replace.elemtype.p1(ptr addrspace(1) %0, /* elemtype=f32.ftz */ i32 8)
  nvvm.tensormap.replace field = elemtype, new_value = #nvvm.tensormap_elemtype<f32.ftz> in %addr : !llvm.ptr<1>

  // CHECK-NEXT: call void @llvm.nvvm.tensormap.replace.elemtype.p1(ptr addrspace(1) %0, /* elemtype=f64 */ i32 9)
  nvvm.tensormap.replace field = elemtype, new_value = #nvvm.tensormap_elemtype<f64> in %addr : !llvm.ptr<1>

  // CHECK-NEXT: call void @llvm.nvvm.tensormap.replace.elemtype.p1(ptr addrspace(1) %0, /* elemtype=bf16 */ i32 10)
  nvvm.tensormap.replace field = elemtype, new_value = #nvvm.tensormap_elemtype<bf16> in %addr : !llvm.ptr<1>

  // CHECK-NEXT: call void @llvm.nvvm.tensormap.replace.elemtype.p1(ptr addrspace(1) %0, /* elemtype=tf32 */ i32 11)
  nvvm.tensormap.replace field = elemtype, new_value = #nvvm.tensormap_elemtype<tf32> in %addr : !llvm.ptr<1>

  // CHECK-NEXT: call void @llvm.nvvm.tensormap.replace.elemtype.p1(ptr addrspace(1) %0, /* elemtype=tf32.ftz */ i32 12)
  nvvm.tensormap.replace field = elemtype, new_value = #nvvm.tensormap_elemtype<tf32.ftz> in %addr : !llvm.ptr<1>

  // CHECK-NEXT: call void @llvm.nvvm.tensormap.replace.elemtype.p1(ptr addrspace(1) %0, /* elemtype=b4x16 */ i32 13)
  nvvm.tensormap.replace field = elemtype, new_value = #nvvm.tensormap_elemtype<b4x16> in %addr : !llvm.ptr<1>

  // CHECK-NEXT: call void @llvm.nvvm.tensormap.replace.elemtype.p1(ptr addrspace(1) %0, /* elemtype=b4x16_p64 */ i32 14)
  nvvm.tensormap.replace field = elemtype, new_value = #nvvm.tensormap_elemtype<b4x16_p64> in %addr : !llvm.ptr<1>

  // CHECK-NEXT: call void @llvm.nvvm.tensormap.replace.elemtype.p1(ptr addrspace(1) %0, /* elemtype=b6x16_p32 */ i32 15)
  nvvm.tensormap.replace field = elemtype, new_value = #nvvm.tensormap_elemtype<b6x16_p32> in %addr : !llvm.ptr<1>

  // CHECK-NEXT: ret void
  llvm.return
  // CHECK-NEXT: }
}

// CHECK-LABEL: define void @tensormap_replace_interleave_layout(ptr addrspace(1) %0) {
llvm.func @tensormap_replace_interleave_layout(%addr : !llvm.ptr<1>) {
  // CHECK-NEXT: call void @llvm.nvvm.tensormap.replace.interleave.layout.p1(ptr addrspace(1) %0, /* interleave_layout=No interleave */ i32 0)
  nvvm.tensormap.replace field = interleave_layout, new_value = #nvvm.tensormap_interleave_layout<no_interleave> in %addr : !llvm.ptr<1>

  // CHECK-NEXT: call void @llvm.nvvm.tensormap.replace.interleave.layout.p1(ptr addrspace(1) %0, /* interleave_layout=16B interleave */ i32 1)
  nvvm.tensormap.replace field = interleave_layout, new_value = #nvvm.tensormap_interleave_layout<b16> in %addr : !llvm.ptr<1>

  // CHECK-NEXT: call void @llvm.nvvm.tensormap.replace.interleave.layout.p1(ptr addrspace(1) %0, /* interleave_layout=32B interleave */ i32 2)
  nvvm.tensormap.replace field = interleave_layout, new_value = #nvvm.tensormap_interleave_layout<b32> in %addr : !llvm.ptr<1>

  // CHECK-NEXT: ret void
  llvm.return
  // CHECK-NEXT: }
}

// CHECK-LABEL: define void @tensormap_replace_swizzle_mode(ptr addrspace(1) %0) {
llvm.func @tensormap_replace_swizzle_mode(%addr : !llvm.ptr<1>) {
  // CHECK-NEXT: call void @llvm.nvvm.tensormap.replace.swizzle.mode.p1(ptr addrspace(1) %0, /* swizzle_mode=No swizzling */ i32 0)
  nvvm.tensormap.replace field = swizzle_mode, new_value = #nvvm.tensormap_swizzle_mode<no_swizzling> in %addr : !llvm.ptr<1>

  // CHECK-NEXT: call void @llvm.nvvm.tensormap.replace.swizzle.mode.p1(ptr addrspace(1) %0, /* swizzle_mode=32B swizzling */ i32 1)
  nvvm.tensormap.replace field = swizzle_mode, new_value = #nvvm.tensormap_swizzle_mode<b32> in %addr : !llvm.ptr<1>

  // CHECK-NEXT: call void @llvm.nvvm.tensormap.replace.swizzle.mode.p1(ptr addrspace(1) %0, /* swizzle_mode=64B swizzling */ i32 2)
  nvvm.tensormap.replace field = swizzle_mode, new_value = #nvvm.tensormap_swizzle_mode<b64> in %addr : !llvm.ptr<1>

  // CHECK-NEXT: call void @llvm.nvvm.tensormap.replace.swizzle.mode.p1(ptr addrspace(1) %0, /* swizzle_mode=128B swizzling */ i32 3)
  nvvm.tensormap.replace field = swizzle_mode, new_value = #nvvm.tensormap_swizzle_mode<b128> in %addr : !llvm.ptr<1>

  // CHECK-NEXT: call void @llvm.nvvm.tensormap.replace.swizzle.mode.p1(ptr addrspace(1) %0, /* swizzle_mode=96B swizzling */ i32 4)
  nvvm.tensormap.replace field = swizzle_mode, new_value = #nvvm.tensormap_swizzle_mode<b96> in %addr : !llvm.ptr<1>

  // CHECK-NEXT: ret void
  llvm.return
  // CHECK-NEXT: }
}

// CHECK-LABEL: define void @tensormap_replace_swizzle_atomicity(ptr addrspace(1) %0) {
llvm.func @tensormap_replace_swizzle_atomicity(%addr : !llvm.ptr<1>) {
  // CHECK-NEXT: call void @llvm.nvvm.tensormap.replace.swizzle.atomicity.p1(ptr addrspace(1) %0, /* swizzle_atomicity=16B */ i32 0)
  nvvm.tensormap.replace field = swizzle_atomicity, new_value = #nvvm.tensormap_swizzle_atomicity<b16> in %addr : !llvm.ptr<1>

  // CHECK-NEXT: call void @llvm.nvvm.tensormap.replace.swizzle.atomicity.p1(ptr addrspace(1) %0, /* swizzle_atomicity=32B */ i32 1)
  nvvm.tensormap.replace field = swizzle_atomicity, new_value = #nvvm.tensormap_swizzle_atomicity<b32> in %addr : !llvm.ptr<1>

  // CHECK-NEXT: call void @llvm.nvvm.tensormap.replace.swizzle.atomicity.p1(ptr addrspace(1) %0, /* swizzle_atomicity=32B + 8B flip */ i32 2)
  nvvm.tensormap.replace field = swizzle_atomicity, new_value = #nvvm.tensormap_swizzle_atomicity<b32_flip_b8> in %addr : !llvm.ptr<1>

  // CHECK-NEXT: call void @llvm.nvvm.tensormap.replace.swizzle.atomicity.p1(ptr addrspace(1) %0, /* swizzle_atomicity=64B */ i32 3)
  nvvm.tensormap.replace field = swizzle_atomicity, new_value = #nvvm.tensormap_swizzle_atomicity<b64> in %addr : !llvm.ptr<1>

  // CHECK-NEXT: ret void
  llvm.return
  // CHECK-NEXT: }
}

// CHECK-LABEL: define void @tensormap_replace_fill_mode(ptr addrspace(1) %0) {
llvm.func @tensormap_replace_fill_mode(%addr : !llvm.ptr<1>) {
  // CHECK-NEXT: call void @llvm.nvvm.tensormap.replace.fill.mode.p1(ptr addrspace(1) %0, /* fill_mode=Zero fill */ i32 0)
  nvvm.tensormap.replace field = fill_mode, new_value = #nvvm.tensormap_fill_mode<zero> in %addr : !llvm.ptr<1>

  // CHECK-NEXT: call void @llvm.nvvm.tensormap.replace.fill.mode.p1(ptr addrspace(1) %0, /* fill_mode=OOB-NaN fill */ i32 1)
  nvvm.tensormap.replace field = fill_mode, new_value = #nvvm.tensormap_fill_mode<oob_nan> in %addr : !llvm.ptr<1>

  // CHECK-NEXT: ret void
  llvm.return
  // CHECK-NEXT: }
}
