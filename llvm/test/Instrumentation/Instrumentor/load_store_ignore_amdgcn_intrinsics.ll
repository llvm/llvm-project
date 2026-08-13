; REQUIRES: amdgpu-registered-target
; RUN: opt < %s -passes=instrumentor -instrumentor-read-config-files=%S/load_store_config.json -S | FileCheck %s

target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

declare noundef nonnull align 4 dereferenceable(64) ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()

define void @instrument_regular_pointer(ptr %dst, ptr %src) {
; CHECK-LABEL: define void @instrument_regular_pointer(
; CHECK-SAME: ptr [[DST:%.*]], ptr [[SRC:%.*]]) {
; CHECK-NEXT: entry:
; CHECK-NEXT: [[SRC_PRE:%.*]] = call ptr @__instrumentor_pre_load(ptr [[SRC]], i32 0, i64 4, i64 4, i32 12, i32 0, i8 1, i8 0)
; CHECK-NEXT: [[SRC_VAL:%.*]] = load i32, ptr [[SRC_PRE]], align 4
; CHECK-NEXT: [[SRC_VAL_ZEXT:%.*]] = zext i32 [[SRC_VAL]] to i64
; CHECK-NEXT: [[SRC_POST:%.*]] = call i64 @__instrumentor_post_load(ptr [[SRC]], i32 0, i64 [[SRC_VAL_ZEXT]], i64 4, i64 4, i32 12, i32 0, i8 1, i8 0)
; CHECK-NEXT: [[STORE_VAL:%.*]] = trunc i64 [[SRC_POST]] to i32
; CHECK-NEXT: [[STORE_VAL_ZEXT:%.*]] = zext i32 [[STORE_VAL]] to i64
; CHECK-NEXT: [[DST_PRE:%.*]] = call ptr @__instrumentor_pre_store(ptr [[DST]], i32 0, i64 [[STORE_VAL_ZEXT]], i64 4, i64 4, i32 12, i32 0, i8 1, i8 0)
; CHECK-NEXT: store i32 [[STORE_VAL]], ptr [[DST_PRE]], align 4
; CHECK-NEXT: call void @__instrumentor_post_store(ptr [[DST]], i32 0, i64 [[STORE_VAL_ZEXT]], i64 4, i64 4, i32 12, i32 0, i8 1, i8 0)
; CHECK-NEXT: ret void
entry:
  %value = load i32, ptr %src, align 4
  store i32 %value, ptr %dst, align 4
  ret void
}

define void @ignore_intrinsic_pointer() {
; CHECK-LABEL: define void @ignore_intrinsic_pointer() {
; CHECK-NEXT: entry:
; CHECK-NEXT: [[DISPATCH:%.*]] = call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
; CHECK-NEXT: [[GENERIC:%.*]] = addrspacecast ptr addrspace(4) [[DISPATCH]] to ptr
; CHECK-NEXT: [[SRC:%.*]] = getelementptr inbounds i8, ptr [[GENERIC]], i64 16
; CHECK-NEXT: [[VALUE:%.*]] = load i32, ptr [[SRC]], align 4
; CHECK-NEXT: [[DST:%.*]] = getelementptr inbounds i8, ptr [[GENERIC]], i64 20
; CHECK-NEXT: store i32 [[VALUE]], ptr [[DST]], align 4
; CHECK-NEXT: ret void
entry:
  %dispatch = call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
  %generic = addrspacecast ptr addrspace(4) %dispatch to ptr
  %src = getelementptr inbounds i8, ptr %generic, i64 16
  %value = load i32, ptr %src, align 4
  %dst = getelementptr inbounds i8, ptr %generic, i64 20
  store i32 %value, ptr %dst, align 4
  ret void
}
