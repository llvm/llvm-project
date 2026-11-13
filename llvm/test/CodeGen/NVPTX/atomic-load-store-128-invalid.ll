; NOTE: Do not autogenerate: these are diagnostic checks.
; RUN: split-file %s %t
; RUN: not llc -mtriple=nvptx64 -mcpu=sm_60 -mattr=+ptx83 -filetype=null %t/unsupported.ll 2>&1 | FileCheck %s --check-prefix=SIZE
; RUN: not llc -mtriple=nvptx64 -mcpu=sm_70 -mattr=+ptx82 -filetype=null %t/unsupported.ll 2>&1 | FileCheck %s --check-prefix=SIZE
; RUN: not llc -mtriple=nvptx64 -mcpu=sm_70 -mattr=+ptx84 -filetype=null %t/unaligned.ll 2>&1 | FileCheck %s --check-prefix=ALIGN
; RUN: not llc -mtriple=nvptx64 -mcpu=sm_80 -mattr=+ptx84 -filetype=null %t/unaligned.ll 2>&1 | FileCheck %s --check-prefix=ALIGN
; RUN: not llc -mtriple=nvptx64 -mcpu=sm_90 -mattr=+ptx84 -filetype=null %t/unaligned.ll 2>&1 | FileCheck %s --check-prefix=ALIGN
; RUN: not llc -mtriple=nvptx64 -mcpu=sm_70 -mattr=+ptx83 -filetype=null %t/system.ll 2>&1 | FileCheck %s --check-prefix=ERROR
; RUN: not llc -mtriple=nvptx64 -mcpu=sm_90 -mattr=+ptx83 -filetype=null %t/system.ll 2>&1 | FileCheck %s --check-prefix=ERROR
; RUN: llc -mtriple=nvptx64 -mcpu=sm_70 -mattr=+ptx84 -verify-machineinstrs -filetype=null %t/system.ll

; Diagnostic checks are handwritten; code-generation checks are generated
; in atomic-load-store-128.ll and atomic-load-store-128-ptx83.ll.

;--- unsupported.ll
; SIZE: error: unsupported atomic load: target supports atomics up to 8 bytes, but this atomic accesses 16 bytes
; SIZE: error: unsupported atomic store: target supports atomics up to 8 bytes, but this atomic accesses 16 bytes
define i128 @load_unsupported(ptr %p) {
  %v = load atomic i128, ptr %p acquire, align 16
  ret i128 %v
}
define void @store_unsupported(ptr %p, i128 %v) {
  store atomic i128 %v, ptr %p release, align 16
  ret void
}

;--- unaligned.ll
; Insufficient alignment is rejected by AtomicExpand before b128 selection.
; ALIGN: error: unsupported atomic load: instruction alignment 8 is smaller than the required 16-byte alignment for this atomic operation
; ALIGN: error: unsupported atomic store: instruction alignment 8 is smaller than the required 16-byte alignment for this atomic operation
define i128 @load_align8(ptr %p) {
  %v = load atomic i128, ptr %p acquire, align 8
  ret i128 %v
}
define void @store_align8(ptr %p, i128 %v) {
  store atomic i128 %v, ptr %p release, align 8
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
