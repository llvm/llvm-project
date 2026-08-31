; RUN: llvm-as -disable-output %s

%S = type { i32, i32 }

define void @runtime_array_nested_access(ptr %src, i32 %index) {
entry:
  %ptr = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype([0 x %S]) %src, i32 %index, i32 1)
  ret void
}

define void @normal_array_access(ptr %src, i32 %index) {
entry:
  %ptr = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype([2 x %S]) %src, i32 %index, i32 1)
  ret void
}

define void @normal_array_constant_index(ptr %src) {
entry:
  %ptr = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype([2 x %S]) %src, i32 1, i32 1)
  ret void
}

define void @struct_access(ptr %src) {
entry:
  %ptr = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype(%S) %src, i32 0)
  ret void
}

define void @nested_array_access(ptr %src) {
entry:
  %ptr = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype([ 3 x [ 2 x i32 ] ]) %src, i32 2, i32 1)
  ret void
}

define void @runtime_array_index(ptr %src) {
entry:
  %ptr = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype([ 0 x i32 ]) %src, i32 1)
  ret void
}

define void @scalar_with_no_index(ptr %src) {
entry:
  %ptr = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype(i32) %src)
  ret void
}

define void @access_64bit(ptr %src) {
entry:
  %ptr = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype([ 0 x i32 ]) %src, i64 1)
  ret void
}

define void @access_8bit(ptr %src) {
entry:
  %ptr = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype([ 0 x i32 ]) %src, i8 1)
  ret void
}
