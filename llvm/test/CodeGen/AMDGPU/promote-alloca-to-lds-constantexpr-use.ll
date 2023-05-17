; RUN: opt -S -disable-promote-alloca-to-vector -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -passes=amdgpu-promote-alloca < %s | FileCheck -check-prefix=IR %s
; RUN: llc -disable-promote-alloca-to-vector -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -amdgpu-enable-lower-module-lds=false < %s | FileCheck -check-prefix=ASM %s

target datalayout = "A5"

@all_lds = internal unnamed_addr addrspace(3) global [16384 x i32] undef, align 4
@some_lds = internal unnamed_addr addrspace(3) global [32 x i32] undef, align 4
@some_dynamic_lds = external hidden addrspace(3) global [0 x i32], align 4

@initializer_user_some = addrspace(1) global i32 ptrtoint (ptr addrspace(3) @some_lds to i32), align 4
@initializer_user_all = addrspace(1) global i32 ptrtoint (ptr addrspace(3) @all_lds to i32), align 4

; This function cannot promote to using LDS because of the size of the
; constant expression use in the function, which was previously not
; detected.
; IR-LABEL: @constant_expression_uses_all_lds(
; IR: alloca

; ASM-LABEL: constant_expression_uses_all_lds:
; ASM: .amdhsa_group_segment_fixed_size 65536
define amdgpu_kernel void @constant_expression_uses_all_lds(ptr addrspace(1) nocapture %out, i32 %idx) #0 {
entry:
  %stack = alloca [4 x i32], align 4, addrspace(5)
  %gep1 = getelementptr inbounds [4 x i32], ptr addrspace(5) %stack, i32 0, i32 1
  %gep2 = getelementptr inbounds [4 x i32], ptr addrspace(5) %stack, i32 0, i32 2
  %gep3 = getelementptr inbounds [4 x i32], ptr addrspace(5) %stack, i32 0, i32 3
  store i32 9, ptr addrspace(5) %stack
  store i32 10, ptr addrspace(5) %gep1
  store i32 99, ptr addrspace(5) %gep2
  store i32 43, ptr addrspace(5) %gep3
  %arrayidx = getelementptr inbounds [4 x i32], ptr addrspace(5) %stack, i32 0, i32 %idx
  %load = load i32, ptr addrspace(5) %arrayidx, align 4
  store i32 %load, ptr addrspace(1) %out

  store volatile i32 ptrtoint (ptr addrspace(3) @all_lds to i32), ptr addrspace(1) undef
  ret void
}

; Has a constant expression use through a single level of constant
; expression, but not enough LDS to block promotion

; IR-LABEL: @constant_expression_uses_some_lds(
; IR-NOT: alloca

; ASM-LABEL: {{^}}constant_expression_uses_some_lds:
; ASM: .amdhsa_group_segment_fixed_size 4224{{$}}
define amdgpu_kernel void @constant_expression_uses_some_lds(ptr addrspace(1) nocapture %out, i32 %idx) #0 {
entry:
  %stack = alloca [4 x i32], align 4, addrspace(5)
  %gep1 = getelementptr inbounds [4 x i32], ptr addrspace(5) %stack, i32 0, i32 1
  %gep2 = getelementptr inbounds [4 x i32], ptr addrspace(5) %stack, i32 0, i32 2
  %gep3 = getelementptr inbounds [4 x i32], ptr addrspace(5) %stack, i32 0, i32 3
  store i32 9, ptr addrspace(5) %stack
  store i32 10, ptr addrspace(5) %gep1
  store i32 99, ptr addrspace(5) %gep2
  store i32 43, ptr addrspace(5) %gep3
  %arrayidx = getelementptr inbounds [4 x i32], ptr addrspace(5) %stack, i32 0, i32 %idx
  %load = load i32, ptr addrspace(5) %arrayidx, align 4
  store i32 %load, ptr addrspace(1) %out
  store volatile i32 ptrtoint (ptr addrspace(3) @some_lds to i32), ptr addrspace(1) undef
  ret void
}

; Has a constant expression use through a single level of constant
; expression, but usage of dynamic LDS should block promotion

; IR-LABEL: @constant_expression_uses_some_dynamic_lds(
; IR: alloca

; ASM-LABEL: {{^}}constant_expression_uses_some_dynamic_lds:
; ASM: .amdhsa_group_segment_fixed_size 0{{$}}
define amdgpu_kernel void @constant_expression_uses_some_dynamic_lds(ptr addrspace(1) nocapture %out, i32 %idx) #0 {
entry:
  %stack = alloca [4 x i32], align 4, addrspace(5)
  %gep1 = getelementptr inbounds [4 x i32], ptr addrspace(5) %stack, i32 0, i32 1
  %gep2 = getelementptr inbounds [4 x i32], ptr addrspace(5) %stack, i32 0, i32 2
  %gep3 = getelementptr inbounds [4 x i32], ptr addrspace(5) %stack, i32 0, i32 3
  store i32 9, ptr addrspace(5) %stack
  store i32 10, ptr addrspace(5) %gep1
  store i32 99, ptr addrspace(5) %gep2
  store i32 43, ptr addrspace(5) %gep3
  %arrayidx = getelementptr inbounds [4 x i32], ptr addrspace(5) %stack, i32 0, i32 %idx
  %load = load i32, ptr addrspace(5) %arrayidx, align 4
  store i32 %load, ptr addrspace(1) %out
  store i32 1234, ptr addrspacecast (ptr addrspace(3) @some_dynamic_lds to ptr), align 4
  ret void
}

declare void @callee(ptr)

; IR-LABEL: @constant_expression_uses_all_lds_multi_level(
; IR: alloca

; ASM-LABEL: {{^}}constant_expression_uses_all_lds_multi_level:
; ASM: .amdhsa_group_segment_fixed_size 65536{{$}}
define amdgpu_kernel void @constant_expression_uses_all_lds_multi_level(ptr addrspace(1) nocapture %out, i32 %idx) #0 {
entry:
  %stack = alloca [4 x i32], align 4, addrspace(5)
  %gep1 = getelementptr inbounds [4 x i32], ptr addrspace(5) %stack, i32 0, i32 1
  %gep2 = getelementptr inbounds [4 x i32], ptr addrspace(5) %stack, i32 0, i32 2
  %gep3 = getelementptr inbounds [4 x i32], ptr addrspace(5) %stack, i32 0, i32 3
  store i32 9, ptr addrspace(5) %stack
  store i32 10, ptr addrspace(5) %gep1
  store i32 99, ptr addrspace(5) %gep2
  store i32 43, ptr addrspace(5) %gep3
  %arrayidx = getelementptr inbounds [4 x i32], ptr addrspace(5) %stack, i32 0, i32 %idx
  %load = load i32, ptr addrspace(5) %arrayidx, align 4
  store i32 %load, ptr addrspace(1) %out
  call void @callee(ptr addrspacecast (ptr addrspace(3) getelementptr inbounds ([16384 x i32], ptr addrspace(3) @all_lds, i32 0, i32 8) to ptr))
  ret void
}

; IR-LABEL: @constant_expression_uses_some_lds_multi_level(
; IR-NOT: alloca
; IR: llvm.amdgcn.workitem.id

; ASM-LABEL: {{^}}constant_expression_uses_some_lds_multi_level:
; ASM: .amdhsa_group_segment_fixed_size 4224{{$}}
define amdgpu_kernel void @constant_expression_uses_some_lds_multi_level(ptr addrspace(1) nocapture %out, i32 %idx) #0 {
entry:
  %stack = alloca [4 x i32], align 4, addrspace(5)
  %gep1 = getelementptr inbounds [4 x i32], ptr addrspace(5) %stack, i32 0, i32 1
  %gep2 = getelementptr inbounds [4 x i32], ptr addrspace(5) %stack, i32 0, i32 2
  %gep3 = getelementptr inbounds [4 x i32], ptr addrspace(5) %stack, i32 0, i32 3
  store i32 9, ptr addrspace(5) %stack
  store i32 10, ptr addrspace(5) %gep1
  store i32 99, ptr addrspace(5) %gep2
  store i32 43, ptr addrspace(5) %gep3
  %arrayidx = getelementptr inbounds [4 x i32], ptr addrspace(5) %stack, i32 0, i32 %idx
  %load = load i32, ptr addrspace(5) %arrayidx, align 4
  store i32 %load, ptr addrspace(1) %out
  call void @callee(ptr addrspacecast (ptr addrspace(3) getelementptr inbounds ([32 x i32], ptr addrspace(3) @some_lds, i32 0, i32 8) to ptr))
  ret void
}

; IR-LABEL: @constant_expression_uses_some_dynamic_lds_multi_level(
; IR: alloca

; ASM-LABEL: {{^}}constant_expression_uses_some_dynamic_lds_multi_level:
; ASM: .amdhsa_group_segment_fixed_size 0{{$}}
define amdgpu_kernel void @constant_expression_uses_some_dynamic_lds_multi_level(ptr addrspace(1) nocapture %out, i32 %idx) #0 {
entry:
  %stack = alloca [4 x i32], align 4, addrspace(5)
  %gep1 = getelementptr inbounds [4 x i32], ptr addrspace(5) %stack, i32 0, i32 1
  %gep2 = getelementptr inbounds [4 x i32], ptr addrspace(5) %stack, i32 0, i32 2
  %gep3 = getelementptr inbounds [4 x i32], ptr addrspace(5) %stack, i32 0, i32 3
  store i32 9, ptr addrspace(5) %stack
  store i32 10, ptr addrspace(5) %gep1
  store i32 99, ptr addrspace(5) %gep2
  store i32 43, ptr addrspace(5) %gep3
  %arrayidx = getelementptr inbounds [4 x i32], ptr addrspace(5) %stack, i32 0, i32 %idx
  %load = load i32, ptr addrspace(5) %arrayidx, align 4
  store i32 %load, ptr addrspace(1) %out
  call void @callee(ptr addrspacecast (ptr addrspace(3) @some_dynamic_lds to ptr))
  ret void
}

; IR-LABEL: @constant_expression_uses_some_lds_global_initializer(
; IR-NOT: alloca
; IR: llvm.amdgcn.workitem.id

; ASM-LABEL: {{^}}constant_expression_uses_some_lds_global_initializer:
; ASM: .amdhsa_group_segment_fixed_size 4096{{$}}
define amdgpu_kernel void @constant_expression_uses_some_lds_global_initializer(ptr addrspace(1) nocapture %out, i32 %idx) #0 {
entry:
  %stack = alloca [4 x i32], align 4, addrspace(5)
  %gep1 = getelementptr inbounds [4 x i32], ptr addrspace(5) %stack, i32 0, i32 1
  %gep2 = getelementptr inbounds [4 x i32], ptr addrspace(5) %stack, i32 0, i32 2
  %gep3 = getelementptr inbounds [4 x i32], ptr addrspace(5) %stack, i32 0, i32 3
  store i32 9, ptr addrspace(5) %stack
  store i32 10, ptr addrspace(5) %gep1
  store i32 99, ptr addrspace(5) %gep2
  store i32 43, ptr addrspace(5) %gep3
  %arrayidx = getelementptr inbounds [4 x i32], ptr addrspace(5) %stack, i32 0, i32 %idx
  %load = load i32, ptr addrspace(5) %arrayidx, align 4
  store i32 %load, ptr addrspace(1) %out

  store volatile i32 ptrtoint (ptr addrspace(1) @initializer_user_some to i32), ptr addrspace(1) undef
  ret void
}

; We can't actually handle LDS initializers in global initializers,
; but this should count as usage.

; IR-LABEL: @constant_expression_uses_all_lds_global_initializer(
; IR: alloca

; ASM-LABEL: {{^}}constant_expression_uses_all_lds_global_initializer:
; ASM: .group_segment_fixed_size: 65536
define amdgpu_kernel void @constant_expression_uses_all_lds_global_initializer(ptr addrspace(1) nocapture %out, i32 %idx) #0 {
entry:
  %stack = alloca [4 x i32], align 4, addrspace(5)
  %gep1 = getelementptr inbounds [4 x i32], ptr addrspace(5) %stack, i32 0, i32 1
  %gep2 = getelementptr inbounds [4 x i32], ptr addrspace(5) %stack, i32 0, i32 2
  %gep3 = getelementptr inbounds [4 x i32], ptr addrspace(5) %stack, i32 0, i32 3
  store i32 9, ptr addrspace(5) %stack
  store i32 10, ptr addrspace(5) %gep1
  store i32 99, ptr addrspace(5) %gep2
  store i32 43, ptr addrspace(5) %gep3
  %arrayidx = getelementptr inbounds [4 x i32], ptr addrspace(5) %stack, i32 0, i32 %idx
  %load = load i32, ptr addrspace(5) %arrayidx, align 4
  store i32 %load, ptr addrspace(1) %out
  store volatile i32 ptrtoint (ptr addrspace(1) @initializer_user_all to i32), ptr addrspace(1) undef
  ret void
}

attributes #0 = { "amdgpu-waves-per-eu"="1,5" "amdgpu-flat-work-group-size"="256,256" }
