; Verify that unsupported atomic operations correctly fail to select: atomic
; load/store of vectors, and atomic load/store/exchange of pointers under the
; logical addressing model.

; RUN: split-file %s %t

; RUN: not llc -O0 -mtriple=spirv64-- %t/load-vector.ll -o /dev/null 2>&1 | FileCheck --check-prefix=FAIL-LOAD-VEC %s

; RUN: not llc -O0 -mtriple=spirv64-- %t/store-vector.ll -o /dev/null 2>&1 | FileCheck --check-prefix=FAIL-STORE-VEC %s

; RUN: not llc -O0 -mtriple=spirv %t/store-ptr-vulkan.ll -o /dev/null 2>&1 | FileCheck --check-prefix=FAIL-STORE-PTR %s

; RUN: not llc -O0 -mtriple=spirv %t/load-ptr-vulkan.ll -o /dev/null 2>&1 | FileCheck --check-prefix=FAIL-LOAD-PTR %s

; RUN: not llc -O0 -mtriple=spirv %t/xchg-ptr-vulkan.ll -o /dev/null 2>&1 | FileCheck --check-prefix=FAIL-XCHG-PTR %s

; FAIL-LOAD-VEC: error:{{.*}}atomic load is only allowed for integer, floating point or pointer types
; FAIL-STORE-VEC: error:{{.*}}atomic store is only allowed for integer or floating point types
; FAIL-STORE-PTR: error:{{.*}}atomic store is only allowed for pointer types for physical addressing model
; FAIL-LOAD-PTR: error:{{.*}}atomic load is only allowed for pointer types for physical addressing model
; FAIL-XCHG-PTR: error:{{.*}}atomic exchange is only allowed for pointer types for physical addressing model

;--- load-vector.ll
define <2 x i32> @load_vector_acquire(ptr addrspace(1) %ptr) {
  %val = load atomic <2 x i32>, ptr addrspace(1) %ptr acquire, align 8
  ret <2 x i32> %val
}

;--- store-vector.ll
define void @store_vector_release(ptr addrspace(1) %ptr, <2 x i32> %val) {
  store atomic <2 x i32> %val, ptr addrspace(1) %ptr release, align 8
  ret void
}

;--- store-ptr-vulkan.ll
define void @store_ptr_release(ptr addrspace(1) %ptr, ptr addrspace(1) %val) {
  store atomic ptr addrspace(1) %val, ptr addrspace(1) %ptr monotonic, align 8
  ret void
}

;--- load-ptr-vulkan.ll
define ptr addrspace(1) @load_ptr_release(ptr addrspace(1) %ptr) {
  %val = load atomic ptr addrspace(1), ptr addrspace(1) %ptr monotonic, align 8
  ret ptr addrspace(1) %val
}

;--- xchg-ptr-vulkan.ll
define ptr addrspace(1) @xchg_ptr_monotonic(ptr addrspace(1) %ptr, ptr addrspace(1) %val) {
  %res = atomicrmw xchg ptr addrspace(1) %ptr, ptr addrspace(1) %val monotonic, align 8
  ret ptr addrspace(1) %res
}
