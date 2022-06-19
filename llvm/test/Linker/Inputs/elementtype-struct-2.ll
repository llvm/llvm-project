%struct = type {i32, i8}

define void @struct_elementtype_2() {
  call %struct* @llvm.preserve.array.access.index.p0s_structs.p0s_structs(%struct* elementtype(%struct) null, i32 0, i32 0)
  ret void
}

declare %struct* @llvm.preserve.array.access.index.p0s_structs.p0s_structs(%struct*, i32, i32)
