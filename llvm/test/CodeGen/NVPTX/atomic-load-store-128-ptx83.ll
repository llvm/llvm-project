; RUN: split-file %s %t
; RUN: llc -mtriple=nvptx64 -mcpu=sm_70 -mattr=+ptx83 -verify-machineinstrs %t/scopes.ll -o - | FileCheck %s --check-prefixes=CHECK,NOCLUSTER
; RUN: llc -mtriple=nvptx64 -mcpu=sm_90 -mattr=+ptx83 -verify-machineinstrs %t/scopes.ll -o - | FileCheck %s --check-prefixes=CHECK,CLUSTER
; RUN: %if ptxas-sm_90 && ptxas-isa-8.3 %{ llc -mtriple=nvptx64 -mcpu=sm_90 -mattr=+ptx83 %t/scopes.ll -o - | %ptxas-verify -arch=sm_90 %}
; RUN: not llc -mtriple=nvptx64 -mcpu=sm_70 -mattr=+ptx83 -filetype=null %t/system.ll 2>&1 | FileCheck %s --check-prefix=ERROR
; RUN: not llc -mtriple=nvptx64 -mcpu=sm_90 -mattr=+ptx83 -filetype=null %t/system.ll 2>&1 | FileCheck %s --check-prefix=ERROR
; RUN: llc -mtriple=nvptx64 -mcpu=sm_70 -mattr=+ptx84 -verify-machineinstrs -filetype=null %t/system.ll

; PTX 8.3 supports b128 loads/stores at narrower scopes. System scope,
; including volatile accesses promoted to system scope, requires PTX 8.4.

;--- scopes.ll

; CHECK-LABEL: load_block(
; CHECK: ld.relaxed.cta.b128
define i128 @load_block(ptr %p) {
  %v = load atomic i128, ptr %p syncscope("block") monotonic, align 16
  ret i128 %v
}

; CHECK-LABEL: load_device(
; CHECK: ld.acquire.gpu.b128
define fp128 @load_device(ptr %p) {
  %v = load atomic fp128, ptr %p syncscope("device") acquire, align 16
  ret fp128 %v
}

; CHECK-LABEL: load_cluster(
; CLUSTER: fence.sc.cluster;
; CLUSTER: ld.acquire.cluster.b128
; NOCLUSTER: fence.sc.cta;
; NOCLUSTER: ld.acquire.cta.b128
define i128 @load_cluster(ptr %p) {
  %v = load atomic i128, ptr %p syncscope("cluster") seq_cst, align 16
  ret i128 %v
}

; CHECK-LABEL: store_block(
; CHECK: st.release.cta.b128
define void @store_block(ptr %p, i128 %v) {
  store atomic i128 %v, ptr %p syncscope("block") release, align 16
  ret void
}

; CHECK-LABEL: store_device(
; CHECK: fence.sc.gpu;
; CHECK: st.relaxed.gpu.b128
define void @store_device(ptr %p, fp128 %v) {
  store atomic fp128 %v, ptr %p syncscope("device") seq_cst, align 16
  ret void
}

; CHECK-LABEL: store_cluster(
; CLUSTER: st.relaxed.cluster.b128
; NOCLUSTER: st.relaxed.cta.b128
define void @store_cluster(ptr %p, i128 %v) {
  store atomic i128 %v, ptr %p syncscope("cluster") monotonic, align 16
  ret void
}

; Generic relaxed volatile operations use .volatile without a scope qualifier.
; CHECK-LABEL: load_volatile_generic(
; CHECK: ld.volatile.b128
define i128 @load_volatile_generic(ptr %p) {
  %v = load atomic volatile i128, ptr %p monotonic, align 16
  ret i128 %v
}

; CHECK-LABEL: store_volatile_generic(
; CHECK: st.volatile.b128
define void @store_volatile_generic(ptr %p, i128 %v) {
  store atomic volatile i128 %v, ptr %p monotonic, align 16
  ret void
}

;--- system.ll

; ERROR: error: {{.*}}in function load_system {{.*}}: NVPTX system scope b128 loads and stores require PTX ISA version 8.4 or later
define i128 @load_system(ptr %p) {
  %v = load atomic i128, ptr %p monotonic, align 16
  ret i128 %v
}

; ERROR: error: {{.*}}in function store_system {{.*}}: NVPTX system scope b128 loads and stores require PTX ISA version 8.4 or later
define void @store_system(ptr %p, i128 %v) {
  store atomic i128 %v, ptr %p monotonic, align 16
  ret void
}

; ERROR: error: {{.*}}in function load_system_fp128 {{.*}}: NVPTX system scope b128 loads and stores require PTX ISA version 8.4 or later
define fp128 @load_system_fp128(ptr %p) {
  %v = load atomic fp128, ptr %p acquire, align 16
  ret fp128 %v
}

; ERROR: error: {{.*}}in function store_system_fp128 {{.*}}: NVPTX system scope b128 loads and stores require PTX ISA version 8.4 or later
define void @store_system_fp128(ptr %p, fp128 %v) {
  store atomic fp128 %v, ptr %p release, align 16
  ret void
}

; ERROR: error: {{.*}}in function load_system_seq_cst {{.*}}: NVPTX system scope b128 loads and stores require PTX ISA version 8.4 or later
define i128 @load_system_seq_cst(ptr %p) {
  %v = load atomic i128, ptr %p seq_cst, align 16
  ret i128 %v
}

; ERROR: error: {{.*}}in function store_system_seq_cst {{.*}}: NVPTX system scope b128 loads and stores require PTX ISA version 8.4 or later
define void @store_system_seq_cst(ptr %p, i128 %v) {
  store atomic i128 %v, ptr %p seq_cst, align 16
  ret void
}

; Global volatile operations use system scope even with a narrower IR scope.
; ERROR: error: {{.*}}in function load_volatile_block {{.*}}: NVPTX system scope b128 loads and stores require PTX ISA version 8.4 or later
define i128 @load_volatile_block(ptr addrspace(1) %p) {
  %v = load atomic volatile i128, ptr addrspace(1) %p syncscope("block") monotonic, align 16
  ret i128 %v
}

; ERROR: error: {{.*}}in function store_volatile_block {{.*}}: NVPTX system scope b128 loads and stores require PTX ISA version 8.4 or later
define void @store_volatile_block(ptr addrspace(1) %p, i128 %v) {
  store atomic volatile i128 %v, ptr addrspace(1) %p syncscope("block") monotonic, align 16
  ret void
}

; ERROR: error: {{.*}}in function load_volatile_acquire {{.*}}: NVPTX system scope b128 loads and stores require PTX ISA version 8.4 or later
define i128 @load_volatile_acquire(ptr addrspace(1) %p) {
  %v = load atomic volatile i128, ptr addrspace(1) %p syncscope("block") acquire, align 16
  ret i128 %v
}

; ERROR: error: {{.*}}in function store_volatile_release {{.*}}: NVPTX system scope b128 loads and stores require PTX ISA version 8.4 or later
define void @store_volatile_release(ptr addrspace(1) %p, i128 %v) {
  store atomic volatile i128 %v, ptr addrspace(1) %p syncscope("block") release, align 16
  ret void
}
