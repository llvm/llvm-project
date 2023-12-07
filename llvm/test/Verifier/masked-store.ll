; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

declare void @llvm.masked.store.v4i32.p0(<4 x i32>, ptr, i32, <4 x i1>)

define void @masked_store(<4 x i1> %mask, ptr %addr, <4 x i32> %val) {
  ; CHECK: masked_store: alignment must be a power of 2
  ; CHECK-NEXT: call void @llvm.masked.store.v4i32.p0(<4 x i32> %val, ptr %addr, i32 3, <4 x i1> %mask)
  call void @llvm.masked.store.v4i32.p0(<4 x i32> %val, ptr %addr, i32 3, <4 x i1> %mask)
  ret void
}
