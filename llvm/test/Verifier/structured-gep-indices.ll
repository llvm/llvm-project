; RUN: llvm-as -disable-output %s

%S = type { i32, i32 }

define void @runtime_array_nested_access(ptr %src, i32 %index) {
entry:
  %ptr = call ptr (ptr, <2 x i32>, ...) @llvm.structured.gep.p0.v2i32(ptr elementtype([0 x %S]) %src, <2 x i32> <i32 4, i32 3>, i32 %index, i32 1)
  ret void
}

define void @normal_array_access(ptr %src, i32 %index) {
entry:
  %ptr = call ptr (ptr, <2 x i32>, ...) @llvm.structured.gep.p0.v2i32(ptr elementtype([2 x %S]) %src, <2 x i32> <i32 5, i32 3>, i32 %index, i32 1)
  ret void
}

define void @normal_array_constant_index(ptr %src) {
entry:
  %ptr = call ptr (ptr, <2 x i32>, ...) @llvm.structured.gep.p0.v2i32(ptr elementtype([2 x %S]) %src, <2 x i32> <i32 5, i32 3>, i32 1, i32 1)
  ret void
}

define void @struct_access(ptr %src) {
entry:
  %ptr = call ptr (ptr, <1 x i32>, ...) @llvm.structured.gep.p0.v1i32(ptr elementtype(%S) %src, <1 x i32> <i32 3>, i32 0)
  ret void
}

define void @nested_array_access(ptr %src) {
entry:
  %ptr = call ptr (ptr, <2 x i32>, ...) @llvm.structured.gep.p0.v2i32(ptr elementtype([ 3 x [ 2 x i32 ] ]) %src, <2 x i32> <i32 5, i32 5>, i32 2, i32 1)
  ret void
}

define void @runtime_array_index(ptr %src) {
entry:
  %ptr = call ptr (ptr, <1 x i32>, ...) @llvm.structured.gep.p0.v1i32(ptr elementtype([ 0 x i32 ]) %src, <1 x i32> <i32 4>, i32 1)
  ret void
}

define void @scalar_with_no_index(ptr %src) {
entry:
  %ptr = call ptr (ptr, <1 x i32>, ...) @llvm.structured.gep.p0.v1i32(ptr elementtype(i32) %src, <1 x i32> zeroinitializer)
  ret void
}

define void @access_64bit(ptr %src) {
entry:
  %ptr = call ptr (ptr, <1 x i32>, ...) @llvm.structured.gep.p0.v1i32(ptr elementtype([ 0 x i32 ]) %src, <1 x i32> <i32 4>, i64 1)
  ret void
}

define void @access_8bit(ptr %src) {
entry:
  %ptr = call ptr (ptr, <1 x i32>, ...) @llvm.structured.gep.p0.v1i32(ptr elementtype([ 0 x i32 ]) %src, <1 x i32> <i32 4>, i8 1)
  ret void
}

define void @array_index_with_unknown_bits(ptr %src, i64 %index) {
entry:
  %ptr = call ptr (ptr, <1 x i32>, ...) @llvm.structured.gep.p0.v1i32(ptr elementtype([ 0 x i32 ]) %src, <1 x i32> <i32 65540>, i64 %index)
  ret void
}

define void @struct_index_with_unknown_bits(ptr %src) {
entry:
  %ptr = call ptr (ptr, <1 x i32>, ...) @llvm.structured.gep.p0.v1i32(ptr elementtype(%S) %src, <1 x i32> <i32 65539>, i32 1)
  ret void
}
