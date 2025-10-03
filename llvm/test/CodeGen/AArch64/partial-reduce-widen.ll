; RUN: llc -mattr=+sve,+dotprod < %s | FileCheck %s

define void @partial_reduce_widen_v1i32_acc_v16i32_vec(ptr %accptr, ptr %resptr, ptr %vecptr) {
  %acc = load <1 x i32>, ptr %accptr
  %vec = load <16 x i32>, ptr %vecptr
  %partial.reduce = call <1 x i32> @llvm.vector.partial.reduce.add(<1 x i32> %acc, <16 x i32> %vec)
  store <1 x i32> %partial.reduce, ptr %resptr
  ret void
}

define void @partial_reduce_widen_v3i32_acc_v12i32_vec(ptr %accptr, ptr %resptr, ptr %vecptr) {
  %acc = load <3 x i32>, ptr %accptr
  %vec = load <12 x i32>, ptr %vecptr
  %partial.reduce = call <3 x i32> @llvm.vector.partial.reduce.add(<3 x i32> %acc, <12 x i32> %vec)
  store <3 x i32> %partial.reduce, ptr %resptr
  ret void
}

define void @partial_reduce_widen_v4i32_acc_v20i32_vec(ptr %accptr, ptr %resptr, ptr %vecptr) {
  %acc = load <1 x i32>, ptr %accptr
  %vec = load <20 x i32>, ptr %vecptr
  %partial.reduce = call <1 x i32> @llvm.vector.partial.reduce.add(<1 x i32> %acc, <20 x i32> %vec)
  store <1 x i32> %partial.reduce, ptr %resptr
  ret void
}
