%struct = type {i32, i8}

define void @struct_elementtype_2() {
  call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype(%struct) null, i32 0, i32 0)
  ret void
}

declare ptr @llvm.preserve.array.access.index.p0.p0(ptr, i32, i32)
