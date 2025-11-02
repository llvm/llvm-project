; RUN: llvm-offload-wrapper --triple=x86-64 -kind=hip %s -o %t.bc
; RUN: llvm-dis %t.bc -o - | FileCheck %s --check-prefix=HIP

;      HIP: @__start_llvm_offload_entries = external hidden constant [0 x %struct.__tgt_offload_entry], section "llvm_offload_entries$OA"
; HIP-NEXT: @__stop_llvm_offload_entries = external hidden constant [0 x %struct.__tgt_offload_entry], section "llvm_offload_entries$OZ"
; HIP-NEXT: @.fatbin_image = internal constant {{.*}}, section ".hip_fatbin"
; HIP-NEXT: @.fatbin_wrapper = internal constant %fatbin_wrapper { i32 1212764230, i32 1, ptr @.fatbin_image, ptr null }, section ".hipFatBinSegment", align 8
; HIP-NEXT: @.hip.binary_handle = internal global ptr null
; HIP-NEXT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 101, ptr @.hip.fatbin_reg, ptr null }]

;      HIP: define internal void @.hip.fatbin_reg() section ".text.startup" {
; HIP-NEXT: entry:
; HIP-NEXT:   %0 = call ptr @__hipRegisterFatBinary(ptr @.fatbin_wrapper)
; HIP-NEXT:   store ptr %0, ptr @.hip.binary_handle, align 8
; HIP-NEXT:   call void @.hip.globals_reg(ptr %0)
; HIP-NEXT:   %1 = call i32 @atexit(ptr @.hip.fatbin_unreg)
; HIP-NEXT:   ret void
; HIP-NEXT: }

;      HIP: define internal void @.hip.fatbin_unreg() section ".text.startup" {
; HIP-NEXT: entry:
; HIP-NEXT:   %0 = load ptr, ptr @.hip.binary_handle, align 8
; HIP-NEXT:   call void @__hipUnregisterFatBinary(ptr %0)
; HIP-NEXT:   ret void
; HIP-NEXT: }

; RUN: llvm-offload-wrapper --triple=x86-64 -kind=cuda %s -o %t.bc
; RUN: llvm-dis %t.bc -o - | FileCheck %s --check-prefix=CUDA

;      CUDA: @__start_llvm_offload_entries = external hidden constant [0 x %struct.__tgt_offload_entry], section "llvm_offload_entries$OA"
; CUDA-NEXT: @__stop_llvm_offload_entries = external hidden constant [0 x %struct.__tgt_offload_entry], section "llvm_offload_entries$OZ"
; CUDA-NEXT: @.fatbin_image = internal constant {{.*}}, section ".nv_fatbin"
; CUDA-NEXT: @.fatbin_wrapper = internal constant %fatbin_wrapper { i32 1180844977, i32 1, ptr @.fatbin_image, ptr null }, section ".nvFatBinSegment", align 8
; CUDA-NEXT: @.cuda.binary_handle = internal global ptr null
; CUDA-NEXT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 101, ptr @.cuda.fatbin_reg, ptr null }]

;      CUDA: define internal void @.cuda.fatbin_reg() section ".text.startup" {
; CUDA-NEXT: entry:
; CUDA-NEXT:   %0 = call ptr @__cudaRegisterFatBinary(ptr @.fatbin_wrapper)
; CUDA-NEXT:   store ptr %0, ptr @.cuda.binary_handle, align 8
; CUDA-NEXT:   call void @.cuda.globals_reg(ptr %0)
; CUDA-NEXT:   call void @__cudaRegisterFatBinaryEnd(ptr %0)
; CUDA-NEXT:   %1 = call i32 @atexit(ptr @.cuda.fatbin_unreg)
; CUDA-NEXT:   ret void
; CUDA-NEXT: }

;      CUDA: define internal void @.cuda.fatbin_unreg() section ".text.startup" {
; CUDA-NEXT: entry:
; CUDA-NEXT:   %0 = load ptr, ptr @.cuda.binary_handle, align 8
; CUDA-NEXT:   call void @__cudaUnregisterFatBinary(ptr %0)
; CUDA-NEXT:   ret void
; CUDA-NEXT: }
