; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

@.str.1 = private unnamed_addr addrspace(1) constant [1 x i8] zeroinitializer, align 1

define linkonce_odr hidden spir_func void @test() {
entry:
; CHECK: %[[#MinusOne:]] = OpConstant %[[#]] 18446744073709551615
; CHECK: %[[#Ptr:]] = OpConvertUToPtr %[[#]]  %[[#MinusOne]]
; CHECK: %[[#PtrCast:]] = OpPtrCastToGeneric %[[#]] %[[#]]
; CHECK: %[[#]] = OpFunctionCall %[[#]] %[[#]] %[[#PtrCast]] %[[#Ptr]]

  %cast = bitcast ptr addrspace(4) inttoptr (i64 -1 to ptr addrspace(4)) to ptr addrspace(4)
  call spir_func void @bar(ptr addrspace(4) addrspacecast (ptr addrspace(1) @.str.1 to ptr addrspace(4)), ptr addrspace(4) %cast)
  ret void
}

define linkonce_odr hidden spir_func void @bar(ptr addrspace(4) %begin, ptr addrspace(4) %end) {
entry:
  ret void
}
