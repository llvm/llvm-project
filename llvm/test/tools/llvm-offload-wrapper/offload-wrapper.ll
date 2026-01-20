; RUN: llvm-offload-wrapper --triple=x86_64-unknown-linux-gnu -kind=openmp %s -o %t.bc
; RUN: llvm-dis %t.bc -o - | FileCheck %s --check-prefix=OMP

;      OMP: @__start_llvm_offload_entries = external hidden constant [0 x %struct.__tgt_offload_entry]
; OMP-NEXT: @__stop_llvm_offload_entries = external hidden constant [0 x %struct.__tgt_offload_entry]
; OMP-NEXT: @__dummy.llvm_offload_entries = internal constant [0 x %struct.__tgt_offload_entry] zeroinitializer, section "llvm_offload_entries", align 8
; OMP-NEXT: @llvm.compiler.used = appending global [1 x ptr] [ptr @__dummy.llvm_offload_entries], section "llvm.metadata"
; OMP-NEXT: @.omp_offloading.device_image = internal unnamed_addr constant [[[SIZE:[0-9]+]] x i8] c"{{.*}}", section ".llvm.offloading", align 8
; OMP-NEXT: @.omp_offloading.device_images = internal unnamed_addr constant [1 x %__tgt_device_image] [%__tgt_device_image { ptr @.omp_offloading.device_image, ptr getelementptr ([[[SIZE]] x i8], ptr @.omp_offloading.device_image, i64 0, i64 [[SIZE]]), ptr @__start_llvm_offload_entries, ptr @__stop_llvm_offload_entries }]
; OMP-NEXT: @.omp_offloading.descriptor = internal constant %__tgt_bin_desc { i32 1, ptr @.omp_offloading.device_images, ptr @__start_llvm_offload_entries, ptr @__stop_llvm_offload_entries }
; OMP-NEXT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 101, ptr @.omp_offloading.descriptor_reg, ptr null }]

;      OMP: define internal void @.omp_offloading.descriptor_reg() section ".text.startup" {
; OMP-NEXT: entry:
; OMP-NEXT:   call void @__tgt_register_lib(ptr @.omp_offloading.descriptor)
; OMP-NEXT:   %0 = call i32 @atexit(ptr @.omp_offloading.descriptor_unreg)
; OMP-NEXT:   ret void
; OMP-NEXT: }

;      OMP: define internal void @.omp_offloading.descriptor_unreg() section ".text.startup" {
; OMP-NEXT: entry:
; OMP-NEXT:   call void @__tgt_unregister_lib(ptr @.omp_offloading.descriptor)
; OMP-NEXT:   ret void
; OMP-NEXT: }

; RUN: llvm-offload-wrapper --triple=x86_64-unknown-linux-gnu -kind=hip %s -o %t.bc
; RUN: llvm-dis %t.bc -o - | FileCheck %s --check-prefix=HIP

;      HIP: @__start_llvm_offload_entries = external hidden constant [0 x %struct.__tgt_offload_entry]
; HIP-NEXT: @__stop_llvm_offload_entries = external hidden constant [0 x %struct.__tgt_offload_entry]
; HIP-NEXT: @__dummy.llvm_offload_entries = internal constant [0 x %struct.__tgt_offload_entry] zeroinitializer, section "llvm_offload_entries", align 8
; HIP-NEXT: @llvm.compiler.used = appending global [1 x ptr] [ptr @__dummy.llvm_offload_entries], section "llvm.metadata"
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

; RUN: llvm-offload-wrapper --triple=x86_64-unknown-linux-gnu -kind=cuda %s -o %t.bc
; RUN: llvm-dis %t.bc -o - | FileCheck %s --check-prefix=CUDA

;      CUDA: @__start_llvm_offload_entries = external hidden constant [0 x %struct.__tgt_offload_entry]
; CUDA-NEXT: @__stop_llvm_offload_entries = external hidden constant [0 x %struct.__tgt_offload_entry]
; CUDA-NEXT: @__dummy.llvm_offload_entries = internal constant [0 x %struct.__tgt_offload_entry] zeroinitializer, section "llvm_offload_entries", align 8
; CUDA-NEXT: @llvm.compiler.used = appending global [1 x ptr] [ptr @__dummy.llvm_offload_entries], section "llvm.metadata"
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
