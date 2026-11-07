; RUN: split-file %s %t
; RUN: not opt -S \
; RUN:   -passes='spirv-emit-intrinsics,function(spirv-legalize-pointer-cast)' \
; RUN:   -mtriple=spirv-unknown-vulkan-compute %t/scalar-offset.ll -o - \
; RUN:   2>&1 | FileCheck %s --check-prefix=OFFSET
; RUN: not opt -S \
; RUN:   -passes='spirv-emit-intrinsics,function(spirv-legalize-pointer-cast)' \
; RUN:   -mtriple=spirv-unknown-vulkan-compute %t/narrow-store.ll -o - \
; RUN:   2>&1 | FileCheck %s --check-prefix=CAST
; RUN: not opt -S \
; RUN:   -passes='spirv-emit-intrinsics,function(spirv-legalize-pointer-cast)' \
; RUN:   -mtriple=spirv-unknown-vulkan-compute %t/wide-load.ll -o - \
; RUN:   2>&1 | FileCheck %s --check-prefix=CAST
; RUN: not opt -S \
; RUN:   -passes='spirv-emit-intrinsics,function(spirv-legalize-pointer-cast)' \
; RUN:   -mtriple=spirv-unknown-vulkan-compute %t/vector-padding.ll -o - \
; RUN:   2>&1 | FileCheck %s --check-prefix=PADDING
; RUN: not opt -S \
; RUN:   -passes='spirv-emit-intrinsics,function(spirv-legalize-pointer-cast)' \
; RUN:   -mtriple=spirv-unknown-vulkan-compute %t/unsized.ll -o - \
; RUN:   2>&1 | FileCheck %s --check-prefix=UNSIZED
; RUN: not opt -S \
; RUN:   -passes='spirv-emit-intrinsics,function(spirv-legalize-pointer-cast)' \
; RUN:   -mtriple=spirv-unknown-vulkan-compute %t/byte-array.ll -o - \
; RUN:   2>&1 | FileCheck %s --check-prefix=CAST
; RUN: not opt -S \
; RUN:   -passes='spirv-emit-intrinsics,function(spirv-legalize-pointer-cast)' \
; RUN:   -mtriple=spirv-unknown-vulkan-compute %t/struct-padding.ll -o - \
; RUN:   2>&1 | FileCheck %s --check-prefix=OFFSET

; OFFSET: LLVM ERROR: Cannot lower a byte offset within a scalar or padding for Logical SPIR-V
; CAST: LLVM ERROR: Cannot lower a pointer cast without equal-width numeric types for Logical SPIR-V
; PADDING: LLVM ERROR: Cannot lower an access to vector padding for Logical SPIR-V
; UNSIZED: LLVM ERROR: Cannot lower byte addressing with an unsized type for Logical SPIR-V

;--- scalar-offset.ll
@Values = external addrspace(3) global [4 x i32]

define void @scalarOffset() {
  store i8 42, ptr addrspace(3) getelementptr inbounds
    (i8, ptr addrspace(3) @Values, i64 1), align 1
  ret void
}

;--- narrow-store.ll
@Values = external addrspace(3) global [4 x i32]

define void @narrowStore() {
  store i16 42, ptr addrspace(3) getelementptr inbounds
    (i8, ptr addrspace(3) @Values, i64 4), align 2
  ret void
}

;--- wide-load.ll
@Values = external addrspace(3) global [4 x i32]

define i64 @wideLoad() {
  %value = load i64, ptr addrspace(3) getelementptr inbounds
    (i8, ptr addrspace(3) @Values, i64 4), align 4
  ret i64 %value
}

;--- vector-padding.ll
target datalayout = "e-p:64:64-i64:64-v96:128"
@Values = external addrspace(3) global [2 x <3 x i32>]

define void @vectorPadding() {
  store i8 42, ptr addrspace(3) getelementptr inbounds
    (i8, ptr addrspace(3) @Values, i64 12), align 1
  ret void
}

;--- unsized.ll
%Opaque = type opaque
@Value = external addrspace(3) global %Opaque

define void @unsized() {
  store i32 42, ptr addrspace(3) getelementptr
    (i8, ptr addrspace(3) @Value, i64 4), align 4
  ret void
}

;--- byte-array.ll
@Values = external addrspace(3) global [8 x i8]

define void @byteArray() {
  store i32 42, ptr addrspace(3) getelementptr inbounds
    (i8, ptr addrspace(3) @Values, i64 4), align 4
  ret void
}

;--- struct-padding.ll
@Value = external addrspace(3) global { i32, i8 }

define void @structPadding() {
  store i8 42, ptr addrspace(3) getelementptr inbounds
    (i8, ptr addrspace(3) @Value, i64 6), align 1
  ret void
}
