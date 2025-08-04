; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -passes=amdgpu-always-inline -amdgpu-enable-lower-module-lds=false %s | FileCheck --check-prefix=ALL %s
; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -passes=amdgpu-always-inline -amdgpu-enable-lower-module-lds=false %s | FileCheck --check-prefix=ALL %s
; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -amdgpu-stress-function-calls -passes=amdgpu-always-inline -amdgpu-enable-lower-module-lds=false %s | FileCheck --check-prefix=ALL %s
; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -amdgpu-stress-function-calls -passes=amdgpu-always-inline -amdgpu-enable-lower-module-lds=false %s | FileCheck --check-prefix=ALL %s

@lds0 = addrspace(3) global i32 poison, align 4
@lds1 = addrspace(3) global [512 x i32] poison, align 4
@nested.lds.address = addrspace(1) global ptr addrspace(3) @lds0, align 4
@gds0 = addrspace(2) global i32 poison, align 4

@alias.lds0 = alias i32, ptr addrspace(3) @lds0
@lds.cycle = addrspace(3) global i32 ptrtoint (ptr addrspace(3) @lds.cycle to i32), align 4


; ALL-LABEL: define i32 @load_lds_simple() #0 {
define i32 @load_lds_simple() {
  %load = load i32, ptr addrspace(3) @lds0, align 4
  ret i32 %load
}

; ALL-LABEL: define i32 @load_gds_simple() #0 {
define i32 @load_gds_simple() {
  %load = load i32, ptr addrspace(2) @gds0, align 4
  ret i32 %load
}

; ALL-LABEL: define i32 @load_lds_const_gep() #0 {
define i32 @load_lds_const_gep() {
  %load = load i32, ptr addrspace(3) getelementptr inbounds ([512 x i32], ptr addrspace(3) @lds1, i64 0, i64 4), align 4
  ret i32 %load
}

; ALL-LABEL: define i32 @load_lds_var_gep(i32 %idx) #0 {
define i32 @load_lds_var_gep(i32 %idx) {
  %gep = getelementptr inbounds [512 x i32], ptr addrspace(3) @lds1, i32 0, i32 %idx
  %load = load i32, ptr addrspace(3) %gep, align 4
  ret i32 %load
}

; ALL-LABEL: define ptr addrspace(3) @load_nested_address(i32 %idx) #0 {
define ptr addrspace(3) @load_nested_address(i32 %idx) {
  %load = load ptr addrspace(3), ptr addrspace(1) @nested.lds.address, align 4
  ret ptr addrspace(3) %load
}

; ALL-LABEL: define i32 @load_lds_alias() #0 {
define i32 @load_lds_alias() {
  %load = load i32, ptr addrspace(3) @alias.lds0, align 4
  ret i32 %load
}

; ALL-LABEL: define i32 @load_lds_cycle() #0 {
define i32 @load_lds_cycle() {
  %load = load i32, ptr addrspace(3) @lds.cycle, align 4
  ret i32 %load
}

; ALL-LABEL: define i1 @icmp_lds_address() #0 {
define i1 @icmp_lds_address() {
  %cmp = icmp eq ptr addrspace(3) @lds0, null
  ret i1 %cmp
}

; ALL-LABEL: define i32 @transitive_call() #0 {
define i32 @transitive_call() {
  %call = call i32 @load_lds_simple()
  ret i32 %call
}

; ALL-LABEL: define i32 @recursive_call_lds(i32 %arg0) #0 {
define i32 @recursive_call_lds(i32 %arg0) {
  %load = load i32, ptr addrspace(3) @lds0, align 4
  %add = add i32 %arg0, %load
  %call = call i32 @recursive_call_lds(i32 %add)
  ret i32 %call
}

; Test we don't break the IR and have both alwaysinline and noinline
; FIXME: We should really not override noinline.

; ALL-LABEL: define i32 @load_lds_simple_noinline() #0 {
define i32 @load_lds_simple_noinline() noinline {
  %load = load i32, ptr addrspace(3) @lds0, align 4
  ret i32 %load
}

; ALL-LABEL: define i32 @recursive_call_lds_noinline(i32 %arg0) #0 {
define i32 @recursive_call_lds_noinline(i32 %arg0) noinline {
  %load = load i32, ptr addrspace(3) @lds0, align 4
  %add = add i32 %arg0, %load
  %call = call i32 @recursive_call_lds(i32 %add)
  ret i32 %call
}

; ALL: attributes #0 = { alwaysinline }
