// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: declare ptr @llvm_ptr_address_space(ptr addrspace(1), ptr addrspace(3))
llvm.func @llvm_ptr_address_space(!ptr.ptr<#llvm.address_space<1>>, !ptr.ptr<#llvm.address_space<3>>) -> !ptr.ptr<#llvm.address_space<0>>

// CHECK-LABEL: define void @llvm_ops_with_ptr_values
// CHECK-SAME:   (ptr %[[ARG:.*]]) {
// CHECK-NEXT:   %[[V0:.*]] = load ptr addrspace(1), ptr %[[ARG]], align 8
// CHECK-NEXT:   store ptr addrspace(1) %[[V0]], ptr %[[ARG]], align 8
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
llvm.func @llvm_ops_with_ptr_values(%arg0: !llvm.ptr) {
  %1 = llvm.load %arg0 : !llvm.ptr -> !ptr.ptr<#llvm.address_space<1>>
  llvm.store %1, %arg0 : !ptr.ptr<#llvm.address_space<1>>, !llvm.ptr
  llvm.return
}

// CHECK-LABEL: define ptr @ptr_add
// CHECK-SAME: (ptr %[[PTR:.*]], i32 %[[OFF:.*]]) {
// CHECK-NEXT:   %[[RES:.*]] = getelementptr i8, ptr %[[PTR]], i32 %[[OFF]]
// CHECK-NEXT:   %[[RES0:.*]] = getelementptr i8, ptr %[[PTR]], i32 %[[OFF]]
// CHECK-NEXT:   %[[RES1:.*]] = getelementptr nusw i8, ptr %[[PTR]], i32 %[[OFF]]
// CHECK-NEXT:   %[[RES2:.*]] = getelementptr nuw i8, ptr %[[PTR]], i32 %[[OFF]]
// CHECK-NEXT:   %[[RES3:.*]] = getelementptr inbounds i8, ptr %[[PTR]], i32 %[[OFF]]
// CHECK-NEXT:   ret ptr %[[RES]]
// CHECK-NEXT: }
llvm.func @ptr_add(%ptr: !ptr.ptr<#llvm.address_space<0>>, %off: i32) -> !ptr.ptr<#llvm.address_space<0>> {
  %res = ptr.ptr_add %ptr, %off : !ptr.ptr<#llvm.address_space<0>>, i32
  %res0 = ptr.ptr_add none %ptr, %off : !ptr.ptr<#llvm.address_space<0>>, i32
  %res1 = ptr.ptr_add nusw %ptr, %off : !ptr.ptr<#llvm.address_space<0>>, i32
  %res2 = ptr.ptr_add nuw %ptr, %off : !ptr.ptr<#llvm.address_space<0>>, i32
  %res3 = ptr.ptr_add inbounds %ptr, %off : !ptr.ptr<#llvm.address_space<0>>, i32
  llvm.return %res : !ptr.ptr<#llvm.address_space<0>>
}

// CHECK-LABEL: define { i32, i32, i32, i32 } @type_offset
// CHECK-NEXT: ret { i32, i32, i32, i32 } { i32 8, i32 1, i32 2, i32 4 }
llvm.func @type_offset(%arg0: !ptr.ptr<#llvm.address_space<0>>) -> !llvm.struct<(i32, i32, i32, i32)> {
  %0 = ptr.type_offset f64 : i32
  %1 = ptr.type_offset i8 : i32
  %2 = ptr.type_offset i16 : i32
  %3 = ptr.type_offset i32 : i32
  %4 = llvm.mlir.poison : !llvm.struct<(i32, i32, i32, i32)>
  %5 = llvm.insertvalue %0, %4[0] : !llvm.struct<(i32, i32, i32, i32)>
  %6 = llvm.insertvalue %1, %5[1] : !llvm.struct<(i32, i32, i32, i32)>
  %7 = llvm.insertvalue %2, %6[2] : !llvm.struct<(i32, i32, i32, i32)>
  %8 = llvm.insertvalue %3, %7[3] : !llvm.struct<(i32, i32, i32, i32)>
  llvm.return %8 : !llvm.struct<(i32, i32, i32, i32)>
}

// CHECK-LABEL: define void @load_ops
// CHECK-SAME: (ptr %[[PTR:.*]]) {
// CHECK-NEXT:   %[[V0:.*]] = load float, ptr %[[PTR]], align 4
// CHECK-NEXT:   %[[V1:.*]] = load volatile float, ptr %[[PTR]], align 4
// CHECK-NEXT:   %[[V2:.*]] = load float, ptr %[[PTR]], align 4, !nontemporal !{{.*}}
// CHECK-NEXT:   %[[V3:.*]] = load float, ptr %[[PTR]], align 4, !invariant.load !{{.*}}
// CHECK-NEXT:   %[[V4:.*]] = load float, ptr %[[PTR]], align 4, !invariant.group !{{.*}}
// CHECK-NEXT:   %[[V5:.*]] = load atomic i64, ptr %[[PTR]] monotonic, align 8
// CHECK-NEXT:   %[[V6:.*]] = load atomic volatile i32, ptr %[[PTR]] syncscope("workgroup") acquire, align 4, !nontemporal !{{.*}}
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
llvm.func @load_ops(%arg0: !ptr.ptr<#llvm.address_space<0>>) {
  %0 = ptr.load %arg0 : !ptr.ptr<#llvm.address_space<0>> -> f32
  %1 = ptr.load volatile %arg0 : !ptr.ptr<#llvm.address_space<0>> -> f32
  %2 = ptr.load %arg0 nontemporal : !ptr.ptr<#llvm.address_space<0>> -> f32
  %3 = ptr.load %arg0 invariant : !ptr.ptr<#llvm.address_space<0>> -> f32
  %4 = ptr.load %arg0 invariant_group : !ptr.ptr<#llvm.address_space<0>> -> f32
  %5 = ptr.load %arg0 atomic monotonic alignment = 8 : !ptr.ptr<#llvm.address_space<0>> -> i64
  %6 = ptr.load volatile %arg0 atomic syncscope("workgroup") acquire nontemporal alignment = 4 : !ptr.ptr<#llvm.address_space<0>> -> i32
  llvm.return
}

// CHECK-LABEL: define void @store_ops
// CHECK-SAME: (ptr %[[PTR:.*]], float %[[ARG1:.*]], i64 %[[ARG2:.*]], i32 %[[ARG3:.*]]) {
// CHECK-NEXT:   store float %[[ARG1]], ptr %[[PTR]], align 4
// CHECK-NEXT:   store volatile float %[[ARG1]], ptr %[[PTR]], align 4
// CHECK-NEXT:   store float %[[ARG1]], ptr %[[PTR]], align 4, !nontemporal !{{.*}}
// CHECK-NEXT:   store float %[[ARG1]], ptr %[[PTR]], align 4, !invariant.group !{{.*}}
// CHECK-NEXT:   store atomic i64 %[[ARG2]], ptr %[[PTR]] monotonic, align 8
// CHECK-NEXT:   store atomic volatile i32 %[[ARG3]], ptr %[[PTR]] syncscope("workgroup") release, align 4, !nontemporal !{{.*}}
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
llvm.func @store_ops(%arg0: !ptr.ptr<#llvm.address_space<0>>, %arg1: f32, %arg2: i64, %arg3: i32) {
  ptr.store %arg1, %arg0 : f32, !ptr.ptr<#llvm.address_space<0>>
  ptr.store volatile %arg1, %arg0 : f32, !ptr.ptr<#llvm.address_space<0>>
  ptr.store %arg1, %arg0 nontemporal : f32, !ptr.ptr<#llvm.address_space<0>>
  ptr.store %arg1, %arg0 invariant_group : f32, !ptr.ptr<#llvm.address_space<0>>
  ptr.store %arg2, %arg0 atomic monotonic alignment = 8 : i64, !ptr.ptr<#llvm.address_space<0>>
  ptr.store volatile %arg3, %arg0 atomic syncscope("workgroup") release nontemporal alignment = 4 : i32, !ptr.ptr<#llvm.address_space<0>>
  llvm.return
}

// CHECK-LABEL: define <4 x float> @gather_ops
// CHECK-SAME: (<4 x ptr> %[[PTRS:.*]], <4 x i1> %[[MASK:.*]], <4 x float> %[[PASSTHROUGH:.*]]) {
// CHECK-NEXT:   %[[V0:.*]] = call <4 x float> @llvm.masked.gather.v4f32.v4p0(<4 x ptr> align 1 %[[PTRS]], <4 x i1> %[[MASK]], <4 x float> %[[PASSTHROUGH]])
// CHECK-NEXT:   %[[V1:.*]] = call <4 x float> @llvm.masked.gather.v4f32.v4p0(<4 x ptr> align 4 %[[PTRS]], <4 x i1> %[[MASK]], <4 x float> %[[PASSTHROUGH]])
// CHECK-NEXT:   ret <4 x float> %[[V0]]
// CHECK-NEXT: }
llvm.func @gather_ops(%ptrs: vector<4x!ptr.ptr<#llvm.address_space<0>>>, %mask: vector<4xi1>, %passthrough: vector<4xf32>) -> vector<4xf32> {
  // Basic gather
  %0 = ptr.gather %ptrs, %mask, %passthrough : vector<4x!ptr.ptr<#llvm.address_space<0>>> -> vector<4xf32>
  // Gather with alignment
  %1 = ptr.gather %ptrs, %mask, %passthrough alignment = 4 : vector<4x!ptr.ptr<#llvm.address_space<0>>> -> vector<4xf32>
  llvm.return %0 : vector<4xf32>
}

// CHECK-LABEL: define <8 x i32> @gather_ops_i32
// CHECK-SAME: (<8 x ptr> %[[PTRS:.*]], <8 x i1> %[[MASK:.*]], <8 x i32> %[[PASSTHROUGH:.*]]) {
// CHECK-NEXT:   %[[V0:.*]] = call <8 x i32> @llvm.masked.gather.v8i32.v8p0(<8 x ptr> align 8 %[[PTRS]], <8 x i1> %[[MASK]], <8 x i32> %[[PASSTHROUGH]])
// CHECK-NEXT:   ret <8 x i32> %[[V0]]
// CHECK-NEXT: }
llvm.func @gather_ops_i32(%ptrs: vector<8x!ptr.ptr<#llvm.address_space<0>>>, %mask: vector<8xi1>, %passthrough: vector<8xi32>) -> vector<8xi32> {
  %0 = ptr.gather %ptrs, %mask, %passthrough alignment = 8 : vector<8x!ptr.ptr<#llvm.address_space<0>>> -> vector<8xi32>
  llvm.return %0 : vector<8xi32>
}

// CHECK-LABEL: define <4 x float> @masked_load_ops
// CHECK-SAME: (ptr %[[PTR:.*]], <4 x i1> %[[MASK:.*]], <4 x float> %[[PASSTHROUGH:.*]]) {
// CHECK-NEXT:   %[[V0:.*]] = call <4 x float> @llvm.masked.load.v4f32.p0(ptr align 1 %[[PTR]], <4 x i1> %[[MASK]], <4 x float> %[[PASSTHROUGH]])
// CHECK-NEXT:   %[[V1:.*]] = call <4 x float> @llvm.masked.load.v4f32.p0(ptr align 16 %[[PTR]], <4 x i1> %[[MASK]], <4 x float> %[[PASSTHROUGH]])
// CHECK-NEXT:   ret <4 x float> %[[V0]]
// CHECK-NEXT: }
llvm.func @masked_load_ops(%ptr: !ptr.ptr<#llvm.address_space<0>>, %mask: vector<4xi1>, %passthrough: vector<4xf32>) -> vector<4xf32> {
  // Basic masked load
  %0 = ptr.masked_load %ptr, %mask, %passthrough : !ptr.ptr<#llvm.address_space<0>> -> vector<4xf32>
  // Masked load with alignment
  %1 = ptr.masked_load %ptr, %mask, %passthrough alignment = 16 : !ptr.ptr<#llvm.address_space<0>> -> vector<4xf32>
  llvm.return %0 : vector<4xf32>
}

// CHECK-LABEL: define <8 x i64> @masked_load_ops_i64
// CHECK-SAME: (ptr %[[PTR:.*]], <8 x i1> %[[MASK:.*]], <8 x i64> %[[PASSTHROUGH:.*]]) {
// CHECK-NEXT:   %[[V0:.*]] = call <8 x i64> @llvm.masked.load.v8i64.p0(ptr align 8 %[[PTR]], <8 x i1> %[[MASK]], <8 x i64> %[[PASSTHROUGH]])
// CHECK-NEXT:   ret <8 x i64> %[[V0]]
// CHECK-NEXT: }
llvm.func @masked_load_ops_i64(%ptr: !ptr.ptr<#llvm.address_space<0>>, %mask: vector<8xi1>, %passthrough: vector<8xi64>) -> vector<8xi64> {
  %0 = ptr.masked_load %ptr, %mask, %passthrough alignment = 8 : !ptr.ptr<#llvm.address_space<0>> -> vector<8xi64>
  llvm.return %0 : vector<8xi64>
}

// CHECK-LABEL: define void @masked_store_ops
// CHECK-SAME: (ptr %[[PTR:.*]], <4 x float> %[[VALUE:.*]], <4 x i1> %[[MASK:.*]]) {
// CHECK-NEXT:   call void @llvm.masked.store.v4f32.p0(<4 x float> %[[VALUE]], ptr align 1 %[[PTR]], <4 x i1> %[[MASK]])
// CHECK-NEXT:   call void @llvm.masked.store.v4f32.p0(<4 x float> %[[VALUE]], ptr align 32 %[[PTR]], <4 x i1> %[[MASK]])
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
llvm.func @masked_store_ops(%ptr: !ptr.ptr<#llvm.address_space<0>>, %value: vector<4xf32>, %mask: vector<4xi1>) {
  // Basic masked store
  ptr.masked_store %value, %ptr, %mask : vector<4xf32>, !ptr.ptr<#llvm.address_space<0>>
  // Masked store with alignment
  ptr.masked_store %value, %ptr, %mask alignment = 32 : vector<4xf32>, !ptr.ptr<#llvm.address_space<0>>
  llvm.return
}

// CHECK-LABEL: define void @masked_store_ops_i16
// CHECK-SAME: (ptr %[[PTR:.*]], <8 x i16> %[[VALUE:.*]], <8 x i1> %[[MASK:.*]]) {
// CHECK-NEXT:   call void @llvm.masked.store.v8i16.p0(<8 x i16> %[[VALUE]], ptr align 4 %[[PTR]], <8 x i1> %[[MASK]])
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
llvm.func @masked_store_ops_i16(%ptr: !ptr.ptr<#llvm.address_space<0>>, %value: vector<8xi16>, %mask: vector<8xi1>) {
  ptr.masked_store %value, %ptr, %mask alignment = 4 : vector<8xi16>, !ptr.ptr<#llvm.address_space<0>>
  llvm.return
}

// CHECK-LABEL: define void @scatter_ops
// CHECK-SAME: (<4 x float> %[[VALUE:.*]], <4 x ptr> %[[PTRS:.*]], <4 x i1> %[[MASK:.*]]) {
// CHECK-NEXT:   call void @llvm.masked.scatter.v4f32.v4p0(<4 x float> %[[VALUE]], <4 x ptr> align 1 %[[PTRS]], <4 x i1> %[[MASK]])
// CHECK-NEXT:   call void @llvm.masked.scatter.v4f32.v4p0(<4 x float> %[[VALUE]], <4 x ptr> align 8 %[[PTRS]], <4 x i1> %[[MASK]])
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
llvm.func @scatter_ops(%value: vector<4xf32>, %ptrs: vector<4x!ptr.ptr<#llvm.address_space<0>>>, %mask: vector<4xi1>) {
  // Basic scatter
  ptr.scatter %value, %ptrs, %mask : vector<4xf32>, vector<4x!ptr.ptr<#llvm.address_space<0>>>
  // Scatter with alignment
  ptr.scatter %value, %ptrs, %mask alignment = 8 : vector<4xf32>, vector<4x!ptr.ptr<#llvm.address_space<0>>>
  llvm.return
}

// CHECK-LABEL: define void @scatter_ops_i64
// CHECK-SAME: (<8 x i64> %[[VALUE:.*]], <8 x ptr> %[[PTRS:.*]], <8 x i1> %[[MASK:.*]]) {
// CHECK-NEXT:   call void @llvm.masked.scatter.v8i64.v8p0(<8 x i64> %[[VALUE]], <8 x ptr> align 16 %[[PTRS]], <8 x i1> %[[MASK]])
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
llvm.func @scatter_ops_i64(%value: vector<8xi64>, %ptrs: vector<8x!ptr.ptr<#llvm.address_space<0>>>, %mask: vector<8xi1>) {
  ptr.scatter %value, %ptrs, %mask alignment = 16 : vector<8xi64>, vector<8x!ptr.ptr<#llvm.address_space<0>>>
  llvm.return
}

// CHECK-LABEL: define void @mixed_masked_ops_address_spaces
// CHECK-SAME: (ptr addrspace(3) %[[PTR_SHARED:.*]], <4 x ptr addrspace(3)> %[[PTRS_SHARED:.*]], <4 x i1> %[[MASK:.*]], <4 x double> %[[VALUE_F64:.*]], <4 x double> %[[PASSTHROUGH_F64:.*]]) {
// CHECK-NEXT:   %[[V0:.*]] = call <4 x double> @llvm.masked.gather.v4f64.v4p3(<4 x ptr addrspace(3)> align 8 %[[PTRS_SHARED]], <4 x i1> %[[MASK]], <4 x double> %[[PASSTHROUGH_F64]])
// CHECK-NEXT:   call void @llvm.masked.scatter.v4f64.v4p3(<4 x double> %[[VALUE_F64]], <4 x ptr addrspace(3)> align 8 %[[PTRS_SHARED]], <4 x i1> %[[MASK]])
// CHECK-NEXT:   %[[V1:.*]] = call <4 x double> @llvm.masked.load.v4f64.p3(ptr addrspace(3) align 8 %[[PTR_SHARED]], <4 x i1> %[[MASK]], <4 x double> %[[PASSTHROUGH_F64]])
// CHECK-NEXT:   call void @llvm.masked.store.v4f64.p3(<4 x double> %[[VALUE_F64]], ptr addrspace(3) align 8 %[[PTR_SHARED]], <4 x i1> %[[MASK]])
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
llvm.func @mixed_masked_ops_address_spaces(%ptr: !ptr.ptr<#llvm.address_space<3>>, %ptrs: vector<4x!ptr.ptr<#llvm.address_space<3>>>,
                                          %mask: vector<4xi1>, %value: vector<4xf64>, %passthrough: vector<4xf64>) {
  // Test with shared memory address space (3) and f64 elements
  %0 = ptr.gather %ptrs, %mask, %passthrough alignment = 8 : vector<4x!ptr.ptr<#llvm.address_space<3>>> -> vector<4xf64>
  ptr.scatter %value, %ptrs, %mask alignment = 8 : vector<4xf64>, vector<4x!ptr.ptr<#llvm.address_space<3>>>
  %1 = ptr.masked_load %ptr, %mask, %passthrough alignment = 8 : !ptr.ptr<#llvm.address_space<3>> -> vector<4xf64>
  ptr.masked_store %value, %ptr, %mask alignment = 8 : vector<4xf64>, !ptr.ptr<#llvm.address_space<3>>
  llvm.return
}

// CHECK-LABEL: define <4 x ptr> @ptr_add_vector
// CHECK-SAME: (<4 x ptr> %[[PTRS:.*]], <4 x i32> %[[OFFSETS:.*]]) {
// CHECK-NEXT:   %[[RES:.*]] = getelementptr i8, <4 x ptr> %[[PTRS]], <4 x i32> %[[OFFSETS]]
// CHECK-NEXT:   ret <4 x ptr> %[[RES]]
// CHECK-NEXT: }
llvm.func @ptr_add_vector(%ptrs: vector<4x!ptr.ptr<#llvm.address_space<0>>>, %offsets: vector<4xi32>) -> vector<4x!ptr.ptr<#llvm.address_space<0>>> {
  %res = ptr.ptr_add %ptrs, %offsets : vector<4x!ptr.ptr<#llvm.address_space<0>>>, vector<4xi32>
  llvm.return %res : vector<4x!ptr.ptr<#llvm.address_space<0>>>
}

// CHECK-LABEL: define <4 x ptr> @ptr_add_scalar_base_vector_offsets
// CHECK-SAME: (ptr %[[PTR:.*]], <4 x i32> %[[OFFSETS:.*]]) {
// CHECK-NEXT:   %[[RES:.*]] = getelementptr i8, ptr %[[PTR]], <4 x i32> %[[OFFSETS]]
// CHECK-NEXT:   ret <4 x ptr> %[[RES]]
// CHECK-NEXT: }
llvm.func @ptr_add_scalar_base_vector_offsets(%ptr: !ptr.ptr<#llvm.address_space<0>>, %offsets: vector<4xi32>) -> vector<4x!ptr.ptr<#llvm.address_space<0>>> {
  %res = ptr.ptr_add %ptr, %offsets : !ptr.ptr<#llvm.address_space<0>>, vector<4xi32>
  llvm.return %res : vector<4x!ptr.ptr<#llvm.address_space<0>>>
}

// CHECK-LABEL: define <4 x ptr> @ptr_add_vector_base_scalar_offset
// CHECK-SAME: (<4 x ptr> %[[PTRS:.*]], i32 %[[OFFSET:.*]]) {
// CHECK-NEXT:   %[[RES:.*]] = getelementptr i8, <4 x ptr> %[[PTRS]], i32 %[[OFFSET]]
// CHECK-NEXT:   ret <4 x ptr> %[[RES]]
// CHECK-NEXT: }
llvm.func @ptr_add_vector_base_scalar_offset(%ptrs: vector<4x!ptr.ptr<#llvm.address_space<0>>>, %offset: i32) -> vector<4x!ptr.ptr<#llvm.address_space<0>>> {
  %res = ptr.ptr_add %ptrs, %offset : vector<4x!ptr.ptr<#llvm.address_space<0>>>, i32
  llvm.return %res : vector<4x!ptr.ptr<#llvm.address_space<0>>>
}

// CHECK-LABEL: declare ptr @nvvm_ptr_address_space(ptr addrspace(1), ptr addrspace(3), ptr addrspace(4), ptr addrspace(5), ptr addrspace(6), ptr addrspace(7))
llvm.func @nvvm_ptr_address_space(
    !ptr.ptr<#nvvm.memory_space<global>>,
    !ptr.ptr<#nvvm.memory_space<shared>>,
    !ptr.ptr<#nvvm.memory_space<constant>>,
    !ptr.ptr<#nvvm.memory_space<local>>,
    !ptr.ptr<#nvvm.memory_space<tensor>>,
    !ptr.ptr<#nvvm.memory_space<shared_cluster>>
  ) -> !ptr.ptr<#nvvm.memory_space<generic>>

// CHECK-LABEL: define void @llvm_ops_with_ptr_nvvm_values
// CHECK-SAME:   (ptr %[[ARG:.*]]) {
// CHECK-NEXT:   %[[V0:.*]] = load ptr addrspace(1), ptr %[[ARG]], align 8
// CHECK-NEXT:   store ptr addrspace(1) %[[V0]], ptr %[[ARG]], align 8
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
llvm.func @llvm_ops_with_ptr_nvvm_values(%arg0: !llvm.ptr) {
  %1 = llvm.load %arg0 : !llvm.ptr -> !ptr.ptr<#nvvm.memory_space<global>>
  llvm.store %1, %arg0 : !ptr.ptr<#nvvm.memory_space<global>>, !llvm.ptr
  llvm.return
}

// CHECK-LABEL: define { ptr, ptr addrspace(1), ptr addrspace(2) } @constant_address_op() {
// CHECK-NEXT: ret { ptr, ptr addrspace(1), ptr addrspace(2) } { ptr null, ptr addrspace(1) inttoptr (i64 4096 to ptr addrspace(1)), ptr addrspace(2) inttoptr (i64 3735928559 to ptr addrspace(2)) }
llvm.func @constant_address_op() ->
    !llvm.struct<(!ptr.ptr<#llvm.address_space<0>>,
                  !ptr.ptr<#llvm.address_space<1>>,
                  !ptr.ptr<#llvm.address_space<2>>)> {
  %0 = ptr.constant #ptr.null : !ptr.ptr<#llvm.address_space<0>>
  %1 = ptr.constant #ptr.address<0x1000> : !ptr.ptr<#llvm.address_space<1>>
  %2 = ptr.constant #ptr.address<3735928559> : !ptr.ptr<#llvm.address_space<2>>
  %3 = llvm.mlir.poison : !llvm.struct<(!ptr.ptr<#llvm.address_space<0>>, !ptr.ptr<#llvm.address_space<1>>, !ptr.ptr<#llvm.address_space<2>>)>
  %4 = llvm.insertvalue %0, %3[0] : !llvm.struct<(!ptr.ptr<#llvm.address_space<0>>, !ptr.ptr<#llvm.address_space<1>>, !ptr.ptr<#llvm.address_space<2>>)>
  %5 = llvm.insertvalue %1, %4[1] : !llvm.struct<(!ptr.ptr<#llvm.address_space<0>>, !ptr.ptr<#llvm.address_space<1>>, !ptr.ptr<#llvm.address_space<2>>)>
  %6 = llvm.insertvalue %2, %5[2] : !llvm.struct<(!ptr.ptr<#llvm.address_space<0>>, !ptr.ptr<#llvm.address_space<1>>, !ptr.ptr<#llvm.address_space<2>>)>
  llvm.return %6 : !llvm.struct<(!ptr.ptr<#llvm.address_space<0>>, !ptr.ptr<#llvm.address_space<1>>, !ptr.ptr<#llvm.address_space<2>>)>
}

// Test gep folders.
// CHECK-LABEL: define ptr @ptr_add_cst() {
// CHECK-NEXT:   ret ptr inttoptr (i64 42 to ptr)
llvm.func @ptr_add_cst() -> !ptr.ptr<#llvm.address_space<0>> {
  %off = llvm.mlir.constant(42 : i32) : i32
  %ptr = ptr.constant #ptr.null : !ptr.ptr<#llvm.address_space<0>>
  %res = ptr.ptr_add %ptr, %off : !ptr.ptr<#llvm.address_space<0>>, i32
  llvm.return %res : !ptr.ptr<#llvm.address_space<0>>
}

// CHECK-LABEL: define i64 @ptr_diff_scalar
// CHECK-SAME: (ptr %[[PTR1:.*]], ptr %[[PTR2:.*]]) {
// CHECK-NEXT:   %[[P1INT:.*]] = ptrtoint ptr %[[PTR1]] to i64
// CHECK-NEXT:   %[[P2INT:.*]] = ptrtoint ptr %[[PTR2]] to i64
// CHECK-NEXT:   %[[DIFF:.*]] = sub i64 %[[P1INT]], %[[P2INT]]
// CHECK-NEXT:   ret i64 %[[DIFF]]
// CHECK-NEXT: }
llvm.func @ptr_diff_scalar(%ptr1: !ptr.ptr<#llvm.address_space<0>>, %ptr2: !ptr.ptr<#llvm.address_space<0>>) -> i64 {
  %diff = ptr.ptr_diff %ptr1, %ptr2 : !ptr.ptr<#llvm.address_space<0>> -> i64
  llvm.return %diff : i64
}

// CHECK-LABEL: define i32 @ptr_diff_scalar_i32
// CHECK-SAME: (ptr %[[PTR1:.*]], ptr %[[PTR2:.*]]) {
// CHECK-NEXT:   %[[P1INT:.*]] = ptrtoint ptr %[[PTR1]] to i64
// CHECK-NEXT:   %[[P2INT:.*]] = ptrtoint ptr %[[PTR2]] to i64
// CHECK-NEXT:   %[[DIFF:.*]] = sub i64 %[[P1INT]], %[[P2INT]]
// CHECK-NEXT:   %[[TRUNC:.*]] = trunc i64 %[[DIFF]] to i32
// CHECK-NEXT:   ret i32 %[[TRUNC]]
// CHECK-NEXT: }
llvm.func @ptr_diff_scalar_i32(%ptr1: !ptr.ptr<#llvm.address_space<0>>, %ptr2: !ptr.ptr<#llvm.address_space<0>>) -> i32 {
  %diff = ptr.ptr_diff %ptr1, %ptr2 : !ptr.ptr<#llvm.address_space<0>> -> i32
  llvm.return %diff : i32
}

// CHECK-LABEL: define <4 x i64> @ptr_diff_vector
// CHECK-SAME: (<4 x ptr> %[[PTRS1:.*]], <4 x ptr> %[[PTRS2:.*]]) {
// CHECK-NEXT:   %[[P1INT:.*]] = ptrtoint <4 x ptr> %[[PTRS1]] to <4 x i64>
// CHECK-NEXT:   %[[P2INT:.*]] = ptrtoint <4 x ptr> %[[PTRS2]] to <4 x i64>
// CHECK-NEXT:   %[[DIFF:.*]] = sub <4 x i64> %[[P1INT]], %[[P2INT]]
// CHECK-NEXT:   ret <4 x i64> %[[DIFF]]
// CHECK-NEXT: }
llvm.func @ptr_diff_vector(%ptrs1: vector<4x!ptr.ptr<#llvm.address_space<0>>>, %ptrs2: vector<4x!ptr.ptr<#llvm.address_space<0>>>) -> vector<4xi64> {
  %diffs = ptr.ptr_diff %ptrs1, %ptrs2 : vector<4x!ptr.ptr<#llvm.address_space<0>>> -> vector<4xi64>
  llvm.return %diffs : vector<4xi64>
}

// CHECK-LABEL: define <8 x i32> @ptr_diff_vector_i32
// CHECK-SAME: (<8 x ptr> %[[PTRS1:.*]], <8 x ptr> %[[PTRS2:.*]]) {
// CHECK-NEXT:   %[[P1INT:.*]] = ptrtoint <8 x ptr> %[[PTRS1]] to <8 x i64>
// CHECK-NEXT:   %[[P2INT:.*]] = ptrtoint <8 x ptr> %[[PTRS2]] to <8 x i64>
// CHECK-NEXT:   %[[DIFF:.*]] = sub <8 x i64> %[[P1INT]], %[[P2INT]]
// CHECK-NEXT:   %[[TRUNC:.*]] = trunc <8 x i64> %[[DIFF]] to <8 x i32>
// CHECK-NEXT:   ret <8 x i32> %[[TRUNC]]
// CHECK-NEXT: }
llvm.func @ptr_diff_vector_i32(%ptrs1: vector<8x!ptr.ptr<#llvm.address_space<0>>>, %ptrs2: vector<8x!ptr.ptr<#llvm.address_space<0>>>) -> vector<8xi32> {
  %diffs = ptr.ptr_diff %ptrs1, %ptrs2 : vector<8x!ptr.ptr<#llvm.address_space<0>>> -> vector<8xi32>
  llvm.return %diffs : vector<8xi32>
}

// CHECK-LABEL: define i64 @ptr_diff_with_constants() {
// CHECK-NEXT:   ret i64 4096
// CHECK-NEXT: }
llvm.func @ptr_diff_with_constants() -> i64 {
  %ptr1 = ptr.constant #ptr.address<0x2000> : !ptr.ptr<#llvm.address_space<0>>
  %ptr2 = ptr.constant #ptr.address<0x1000> : !ptr.ptr<#llvm.address_space<0>>
  %diff = ptr.ptr_diff %ptr1, %ptr2 : !ptr.ptr<#llvm.address_space<0>> -> i64
  llvm.return %diff : i64
}

// CHECK-LABEL: define i64 @ptr_diff_with_flags_nsw
// CHECK-SAME: (ptr %[[PTR1:.*]], ptr %[[PTR2:.*]]) {
// CHECK-NEXT:   %[[P1INT:.*]] = ptrtoint ptr %[[PTR1]] to i64
// CHECK-NEXT:   %[[P2INT:.*]] = ptrtoint ptr %[[PTR2]] to i64
// CHECK-NEXT:   %[[DIFF:.*]] = sub nsw i64 %[[P1INT]], %[[P2INT]]
// CHECK-NEXT:   ret i64 %[[DIFF]]
// CHECK-NEXT: }
llvm.func @ptr_diff_with_flags_nsw(%ptr1: !ptr.ptr<#llvm.address_space<0>>, %ptr2: !ptr.ptr<#llvm.address_space<0>>) -> i64 {
  %diff = ptr.ptr_diff nsw %ptr1, %ptr2 : !ptr.ptr<#llvm.address_space<0>> -> i64
  llvm.return %diff : i64
}

// CHECK-LABEL: define i64 @ptr_diff_with_flags_nuw
// CHECK-SAME: (ptr %[[PTR1:.*]], ptr %[[PTR2:.*]]) {
// CHECK-NEXT:   %[[P1INT:.*]] = ptrtoint ptr %[[PTR1]] to i64
// CHECK-NEXT:   %[[P2INT:.*]] = ptrtoint ptr %[[PTR2]] to i64
// CHECK-NEXT:   %[[DIFF:.*]] = sub nuw i64 %[[P1INT]], %[[P2INT]]
// CHECK-NEXT:   ret i64 %[[DIFF]]
// CHECK-NEXT: }
llvm.func @ptr_diff_with_flags_nuw(%ptr1: !ptr.ptr<#llvm.address_space<0>>, %ptr2: !ptr.ptr<#llvm.address_space<0>>) -> i64 {
  %diff = ptr.ptr_diff nuw %ptr1, %ptr2 : !ptr.ptr<#llvm.address_space<0>> -> i64
  llvm.return %diff : i64
}

// CHECK-LABEL: define i64 @ptr_diff_with_flags_nsw_nuw
// CHECK-SAME: (ptr %[[PTR1:.*]], ptr %[[PTR2:.*]]) {
// CHECK-NEXT:   %[[P1INT:.*]] = ptrtoint ptr %[[PTR1]] to i64
// CHECK-NEXT:   %[[P2INT:.*]] = ptrtoint ptr %[[PTR2]] to i64
// CHECK-NEXT:   %[[DIFF:.*]] = sub nuw nsw i64 %[[P1INT]], %[[P2INT]]
// CHECK-NEXT:   ret i64 %[[DIFF]]
// CHECK-NEXT: }
llvm.func @ptr_diff_with_flags_nsw_nuw(%ptr1: !ptr.ptr<#llvm.address_space<0>>, %ptr2: !ptr.ptr<#llvm.address_space<0>>) -> i64 {
  %diff = ptr.ptr_diff nsw | nuw %ptr1, %ptr2 : !ptr.ptr<#llvm.address_space<0>> -> i64
  llvm.return %diff : i64
}
