; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

declare void @llvm.masked.store.v4i32.p0v4i32(<4 x i32>, <4 x i32>*, i32, <4 x i1>)

define void @masked_store(<4 x i1> %mask, <4 x i32>* %addr, <4 x i32> %val) {
  ; CHECK: masked_store: alignment must be a power of 2
  ; CHECK-NEXT: call void @llvm.masked.store.v4i32.p0v4i32(<4 x i32> %val, <4 x i32>* %addr, i32 3, <4 x i1> %mask)
  call void @llvm.masked.store.v4i32.p0v4i32(<4 x i32> %val, <4 x i32>* %addr, i32 3, <4 x i1> %mask)
  ret void
}
