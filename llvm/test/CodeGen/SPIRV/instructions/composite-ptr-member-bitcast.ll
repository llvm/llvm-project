; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - | FileCheck %s --check-prefix=UNTYPED
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - -filetype=obj | spirv-val %}

; OpCompositeExtract results and OpCompositeInsert objects must match the
; composite member's type. Typed pointers need OpBitcast where pointee types
; differ; untyped pointers need no casts.

; UNTYPED-NOT: OpBitcast

; CHECK-DAG: %[[#Char:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#Int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#CharPtr:]] = OpTypePointer Function %[[#Char]]
; CHECK-DAG: %[[#IntPtr:]] = OpTypePointer Function %[[#Int]]
; CHECK-DAG: %[[#Struct:]] = OpTypeStruct %[[#CharPtr]] %[[#Char]]
; CHECK-DAG: %[[#GlobalIntPtr:]] = OpTypePointer CrossWorkgroup %[[#Int]]
; CHECK-DAG: %[[#GlobalCharPtr:]] = OpTypePointer CrossWorkgroup %[[#Char]]
; CHECK-DAG: %[[#GlobalStruct:]] = OpTypeStruct %[[#GlobalIntPtr]] %[[#Char]]
; CHECK-DAG: %[[#Array:]] = OpTypeArray %[[#CharPtr]] %[[#]]

define internal { ptr, i8 } @ret_aggr() {
  ret { ptr, i8 } { ptr null, i8 1 }
}

; CHECK: OpFunction
; CHECK: %[[#Aggr:]] = OpFunctionCall %[[#Struct]]
; CHECK: %[[#Member:]] = OpCompositeExtract %[[#CharPtr]] %[[#Aggr]] 0
; CHECK: %[[#Ptr:]] = OpBitcast %[[#IntPtr]] %[[#Member]]
; CHECK: OpLoad %[[#Int]] %[[#Ptr]]
define spir_kernel void @extract(ptr addrspace(1) %out) {
  %r = call { ptr, i8 } @ret_aggr()
  %p = extractvalue { ptr, i8 } %r, 0
  %v = load i32, ptr %p
  store i32 %v, ptr addrspace(1) %out
  ret void
}

define internal { i32, { ptr, i8 } } @ret_nested_aggr() {
  ret { i32, { ptr, i8 } } { i32 0, { ptr, i8 } { ptr null, i8 1 } }
}

; CHECK: OpFunction
; CHECK: %[[#Aggr:]] = OpFunctionCall %[[#]]
; CHECK: %[[#Member:]] = OpCompositeExtract %[[#CharPtr]] %[[#Aggr]] 1 0
; CHECK: %[[#Ptr:]] = OpBitcast %[[#IntPtr]] %[[#Member]]
; CHECK: OpLoad %[[#Int]] %[[#Ptr]]
define spir_kernel void @extract_nested(ptr addrspace(1) %out) {
  %r = call { i32, { ptr, i8 } } @ret_nested_aggr()
  %p = extractvalue { i32, { ptr, i8 } } %r, 1, 0
  %v = load i32, ptr %p
  store i32 %v, ptr addrspace(1) %out
  ret void
}

define internal i8 @take_aggr({ ptr, i8 } %a) {
  %p = extractvalue { ptr, i8 } %a, 0
  %v = load i8, ptr %p
  ret i8 %v
}

; CHECK: OpFunction
; CHECK: %[[#Var:]] = OpVariable %[[#IntPtr]] Function
; CHECK: %[[#Cast:]] = OpBitcast %[[#CharPtr]] %[[#Var]]
; CHECK: OpCompositeInsert %[[#Struct]] %[[#Cast]] %[[#]] 0
define spir_kernel void @insert(ptr addrspace(1) %out) {
  %x = alloca i32
  store i32 7, ptr %x
  %a = insertvalue { ptr, i8 } poison, ptr %x, 0
  %b = insertvalue { ptr, i8 } %a, i8 1, 1
  %r = call i8 @take_aggr({ ptr, i8 } %b)
  store i8 %r, ptr addrspace(1) %out
  ret void
}

; A constant aggregate records the pointee type of its pointer members, so
; the extracted pointer only needs a bitcast when the use disagrees.
@g = addrspace(1) global i32 42

; CHECK: OpFunction
; CHECK: %[[#Member:]] = OpCompositeExtract %[[#GlobalIntPtr]] %[[#]] 0
; CHECK-NOT: OpBitcast
; CHECK: OpLoad %[[#Int]] %[[#Member]]
define spir_kernel void @extract_const(ptr addrspace(1) %out) {
  %p = extractvalue { ptr addrspace(1), i8 } { ptr addrspace(1) @g, i8 1 }, 0
  %v = load i32, ptr addrspace(1) %p
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; CHECK: OpFunction
; CHECK: %[[#Member:]] = OpCompositeExtract %[[#GlobalIntPtr]] %[[#]] 0
; CHECK: %[[#Ptr:]] = OpBitcast %[[#GlobalCharPtr]] %[[#Member]]
; CHECK: OpLoad %[[#Char]] %[[#Ptr]]
define spir_kernel void @extract_const_mismatch(ptr addrspace(1) %out) {
  %p = extractvalue { ptr addrspace(1), i8 } { ptr addrspace(1) @g, i8 1 }, 0
  %v = load i8, ptr addrspace(1) %p
  store i8 %v, ptr addrspace(1) %out
  ret void
}

; Arrays of pointers behave like structs.

define internal [2 x ptr] @ret_arr() {
  ret [2 x ptr] [ptr null, ptr null]
}

; CHECK: OpFunction
; CHECK: %[[#Aggr:]] = OpFunctionCall %[[#Array]]
; CHECK: %[[#Member:]] = OpCompositeExtract %[[#CharPtr]] %[[#Aggr]] 1
; CHECK: %[[#Ptr:]] = OpBitcast %[[#IntPtr]] %[[#Member]]
; CHECK: OpLoad %[[#Int]] %[[#Ptr]]
define spir_kernel void @extract_array(ptr addrspace(1) %out) {
  %r = call [2 x ptr] @ret_arr()
  %p = extractvalue [2 x ptr] %r, 1
  %v = load i32, ptr %p
  store i32 %v, ptr addrspace(1) %out
  ret void
}

define internal i8 @take_arr([2 x ptr] %a) {
  %p = extractvalue [2 x ptr] %a, 1
  %v = load i8, ptr %p
  ret i8 %v
}

; CHECK: OpFunction
; CHECK: %[[#Var:]] = OpVariable %[[#IntPtr]] Function
; CHECK: %[[#Cast:]] = OpBitcast %[[#CharPtr]] %[[#Var]]
; CHECK: OpCompositeInsert %[[#Array]] %[[#Cast]] %[[#]] 1
define spir_kernel void @insert_array(ptr addrspace(1) %out) {
  %x = alloca i32
  store i32 7, ptr %x
  %a = insertvalue [2 x ptr] poison, ptr %x, 1
  %r = call i8 @take_arr([2 x ptr] %a)
  store i8 %r, ptr addrspace(1) %out
  ret void
}
