; RUN: opt -mtriple=amdgcn-amd-amdhsa -load-store-vectorizer -S -o - %s | FileCheck %s

; Checks that we don't merge loads/stores of types smaller than one
; byte, or vectors with elements smaller than one byte.

%struct.foo = type { i32, i8 }

declare void @use_i1(i1)
declare void @use_i2(i2)
declare void @use_i8(i8)
declare void @use_foo(%struct.foo)
declare void @use_v2i2(<2 x i2>)
declare void @use_v4i2(<4 x i2>)
declare void @use_v2i9(<2 x i9>)

; CHECK-LABEL: @merge_store_2_constants_i1(
; CHECK: store i1
; CHECK: store i1
define void @merge_store_2_constants_i1(i1 addrspace(1)* %out) #0 {
  %out.gep.1 = getelementptr i1, i1 addrspace(1)* %out, i32 1
  store i1 true, i1 addrspace(1)* %out.gep.1
  store i1 false, i1 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: @merge_store_2_constants_i2(
; CHECK: store i2 1
; CHECK: store i2 -1
define void @merge_store_2_constants_i2(i2 addrspace(1)* %out) #0 {
  %out.gep.1 = getelementptr i2, i2 addrspace(1)* %out, i32 1
  store i2 1, i2 addrspace(1)* %out.gep.1
  store i2 -1, i2 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: @merge_different_store_sizes_i1_i8(
; CHECK: store i1 true
; CHECK: store i8 123
define void @merge_different_store_sizes_i1_i8(i8 addrspace(1)* %out) #0 {
  %out.i1 = bitcast i8 addrspace(1)* %out to i1 addrspace(1)*
  %out.gep.1 = getelementptr i8, i8 addrspace(1)* %out, i32 1
  store i1 true, i1 addrspace(1)* %out.i1
  store i8 123, i8 addrspace(1)* %out.gep.1
  ret void
}

; CHECK-LABEL: @merge_different_store_sizes_i8_i1(
; CHECK: store i8 123
; CHECK: store i1 true
define void @merge_different_store_sizes_i8_i1(i1 addrspace(1)* %out) #0 {
  %out.i8 = bitcast i1 addrspace(1)* %out to i8 addrspace(1)*
  %out.gep.1 = getelementptr i8, i8 addrspace(1)* %out.i8, i32 1
  store i8 123, i8 addrspace(1)* %out.gep.1
  store i1 true, i1 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: @merge_store_2_constant_structs(
; CHECK: store %struct.foo
; CHECK: store %struct.foo
define void @merge_store_2_constant_structs(%struct.foo addrspace(1)* %out) #0 {
  %out.gep.1 = getelementptr %struct.foo, %struct.foo addrspace(1)* %out, i32 1
  store %struct.foo { i32 12, i8 3 }, %struct.foo addrspace(1)* %out.gep.1
  store %struct.foo { i32 92, i8 9 }, %struct.foo addrspace(1)* %out
  ret void
}

; sub-byte element size
; CHECK-LABEL: @merge_store_2_constants_v2i2(
; CHECK: store <2 x i2>
; CHECK: store <2 x i2>
define void @merge_store_2_constants_v2i2(<2 x i2> addrspace(1)* %out) #0 {
  %out.gep.1 = getelementptr <2 x i2>, <2 x i2> addrspace(1)* %out, i32 1
  store <2 x i2> <i2 1, i2 -1>, <2 x i2> addrspace(1)* %out.gep.1
  store <2 x i2> <i2 -1, i2 1>, <2 x i2> addrspace(1)* %out
  ret void
}

; sub-byte element size but byte size

; CHECK-LABEL: @merge_store_2_constants_v4i2(
; CHECK: store <4 x i2>
; CHECK: store <4 x i2>
define void @merge_store_2_constants_v4i2(<4 x i2> addrspace(1)* %out) #0 {
  %out.gep.1 = getelementptr <4 x i2>, <4 x i2> addrspace(1)* %out, i32 1
  store <4 x i2> <i2 1, i2 -1, i2 1, i2 -1>, <4 x i2> addrspace(1)* %out.gep.1
  store <4 x i2> <i2 -1, i2 1, i2 -1, i2 1>, <4 x i2> addrspace(1)* %out
  ret void
}

; CHECK-LABEL: @merge_load_2_constants_i1(
; CHECK: load i1
; CHECK: load i1
define void @merge_load_2_constants_i1(i1 addrspace(1)* %out) #0 {
  %out.gep.1 = getelementptr i1, i1 addrspace(1)* %out, i32 1
  %x = load i1, i1 addrspace(1)* %out.gep.1
  %y = load i1, i1 addrspace(1)* %out
  call void @use_i1(i1 %x)
  call void @use_i1(i1 %y)
  ret void
}

; CHECK-LABEL: @merge_load_2_constants_i2(
; CHECK: load i2
; CHECK: load i2
define void @merge_load_2_constants_i2(i2 addrspace(1)* %out) #0 {
  %out.gep.1 = getelementptr i2, i2 addrspace(1)* %out, i32 1
  %x = load i2, i2 addrspace(1)* %out.gep.1
  %y = load i2, i2 addrspace(1)* %out
  call void @use_i2(i2 %x)
  call void @use_i2(i2 %y)
  ret void
}

; CHECK-LABEL: @merge_different_load_sizes_i1_i8(
; CHECK: load i1
; CHECK: load i8
define void @merge_different_load_sizes_i1_i8(i8 addrspace(1)* %out) #0 {
  %out.i1 = bitcast i8 addrspace(1)* %out to i1 addrspace(1)*
  %out.gep.1 = getelementptr i8, i8 addrspace(1)* %out, i32 1
  %x = load i1, i1 addrspace(1)* %out.i1
  %y = load i8, i8 addrspace(1)* %out.gep.1
  call void @use_i1(i1 %x)
  call void @use_i8(i8 %y)
  ret void
}

; CHECK-LABEL: @merge_different_load_sizes_i8_i1(
; CHECK: load i8
; CHECK: load i1
define void @merge_different_load_sizes_i8_i1(i1 addrspace(1)* %out) #0 {
  %out.i8 = bitcast i1 addrspace(1)* %out to i8 addrspace(1)*
  %out.gep.1 = getelementptr i8, i8 addrspace(1)* %out.i8, i32 1
  %x = load i8, i8 addrspace(1)* %out.gep.1
  %y = load i1, i1 addrspace(1)* %out
  call void @use_i8(i8 %x)
  call void @use_i1(i1 %y)
  ret void
}

; CHECK-LABEL: @merge_load_2_constant_structs(
; CHECK: load %struct.foo
; CHECK: load %struct.foo
define void @merge_load_2_constant_structs(%struct.foo addrspace(1)* %out) #0 {
  %out.gep.1 = getelementptr %struct.foo, %struct.foo addrspace(1)* %out, i32 1
  %x = load %struct.foo, %struct.foo addrspace(1)* %out.gep.1
  %y = load %struct.foo, %struct.foo addrspace(1)* %out
  call void @use_foo(%struct.foo %x)
  call void @use_foo(%struct.foo %y)
  ret void
}

; CHECK-LABEL: @merge_load_2_constants_v2i2(
; CHECK: load <2 x i2>
; CHECK: load <2 x i2>
define void @merge_load_2_constants_v2i2(<2 x i2> addrspace(1)* %out) #0 {
  %out.gep.1 = getelementptr <2 x i2>, <2 x i2> addrspace(1)* %out, i32 1
  %x = load <2 x i2>, <2 x i2> addrspace(1)* %out.gep.1
  %y = load <2 x i2>, <2 x i2> addrspace(1)* %out
  call void @use_v2i2(<2 x i2> %x)
  call void @use_v2i2(<2 x i2> %y)
  ret void
}

; CHECK-LABEL: @merge_load_2_constants_v4i2(
; CHECK: load <4 x i2>
; CHECK: load <4 x i2>
define void @merge_load_2_constants_v4i2(<4 x i2> addrspace(1)* %out) #0 {
  %out.gep.1 = getelementptr <4 x i2>, <4 x i2> addrspace(1)* %out, i32 1
  %x = load <4 x i2>, <4 x i2> addrspace(1)* %out.gep.1
  %y = load <4 x i2>, <4 x i2> addrspace(1)* %out
  call void @use_v4i2(<4 x i2> %x)
  call void @use_v4i2(<4 x i2> %y)
  ret void
}

; CHECK-LABEL: @merge_store_2_constants_i9(
; CHECK: store i9 3
; CHECK: store i9 -5
define void @merge_store_2_constants_i9(i9 addrspace(1)* %out) #0 {
  %out.gep.1 = getelementptr i9, i9 addrspace(1)* %out, i32 1
  store i9 3, i9 addrspace(1)* %out.gep.1
  store i9 -5, i9 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: @merge_load_2_constants_v2i9(
; CHECK: load <2 x i9>
; CHECK: load <2 x i9>
define void @merge_load_2_constants_v2i9(<2 x i9> addrspace(1)* %out) #0 {
  %out.gep.1 = getelementptr <2 x i9>, <2 x i9> addrspace(1)* %out, i32 1
  %x = load <2 x i9>, <2 x i9> addrspace(1)* %out.gep.1
  %y = load <2 x i9>, <2 x i9> addrspace(1)* %out
  call void @use_v2i9(<2 x i9> %x)
  call void @use_v2i9(<2 x i9> %y)
  ret void
}

attributes #0 = { nounwind }
