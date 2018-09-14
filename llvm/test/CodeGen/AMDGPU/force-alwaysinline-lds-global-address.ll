; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -amdgpu-function-calls -amdgpu-always-inline %s | FileCheck -check-prefixes=CALLS-ENABLED,ALL %s
; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -amdgpu-function-calls -amdgpu-stress-function-calls -amdgpu-always-inline %s | FileCheck -check-prefixes=STRESS-CALLS,ALL %s

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5"

@lds0 = addrspace(3) global i32 undef, align 4
@lds1 = addrspace(3) global [512 x i32] undef, align 4
@nested.lds.address = addrspace(1) global i32 addrspace(3)* @lds0, align 4
@gds0 = addrspace(2) global i32 undef, align 4

@alias.lds0 = alias i32, i32 addrspace(3)* @lds0
@lds.cycle = addrspace(3) global i32 ptrtoint (i32 addrspace(3)* @lds.cycle to i32), align 4


; ALL-LABEL: define i32 @load_lds_simple() #0 {
define i32 @load_lds_simple() {
  %load = load i32, i32 addrspace(3)* @lds0, align 4
  ret i32 %load
}

; ALL-LABEL: define i32 @load_gds_simple() #0 {
define i32 @load_gds_simple() {
  %load = load i32, i32 addrspace(2)* @gds0, align 4
  ret i32 %load
}

; ALL-LABEL: define i32 @load_lds_const_gep() #0 {
define i32 @load_lds_const_gep() {
  %load = load i32, i32 addrspace(3)* getelementptr inbounds ([512 x i32], [512 x i32] addrspace(3)* @lds1, i64 0, i64 4), align 4
  ret i32 %load
}

; ALL-LABEL: define i32 @load_lds_var_gep(i32 %idx) #0 {
define i32 @load_lds_var_gep(i32 %idx) {
  %gep = getelementptr inbounds [512 x i32], [512 x i32] addrspace(3)* @lds1, i32 0, i32 %idx
  %load = load i32, i32 addrspace(3)* %gep, align 4
  ret i32 %load
}

; ALL-LABEL: define i32 addrspace(3)* @load_nested_address(i32 %idx) #0 {
define i32 addrspace(3)* @load_nested_address(i32 %idx) {
  %load = load i32 addrspace(3)*, i32 addrspace(3)* addrspace(1)* @nested.lds.address, align 4
  ret i32 addrspace(3)* %load
}

; ALL-LABEL: define i32 @load_lds_alias() #0 {
define i32 @load_lds_alias() {
  %load = load i32, i32 addrspace(3)* @alias.lds0, align 4
  ret i32 %load
}

; ALL-LABEL: define i32 @load_lds_cycle() #0 {
define i32 @load_lds_cycle() {
  %load = load i32, i32 addrspace(3)* @lds.cycle, align 4
  ret i32 %load
}

; ALL-LABEL: define i1 @icmp_lds_address() #0 {
define i1 @icmp_lds_address() {
  ret i1 icmp eq (i32 addrspace(3)* @lds0, i32 addrspace(3)* null)
}

; ALL-LABEL: define i32 @transitive_call() #0 {
define i32 @transitive_call() {
  %call = call i32 @load_lds_simple()
  ret i32 %call
}

; ALL-LABEL: define i32 @recursive_call_lds(i32 %arg0) #0 {
define i32 @recursive_call_lds(i32 %arg0) {
  %load = load i32, i32 addrspace(3)* @lds0, align 4
  %add = add i32 %arg0, %load
  %call = call i32 @recursive_call_lds(i32 %add)
  ret i32 %call
}

; ALL: attributes #0 = { alwaysinline }
