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
