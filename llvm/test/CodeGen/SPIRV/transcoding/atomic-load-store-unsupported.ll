; Verify that atomic load/store of vectors and volatile atomic load/store
; correctly fail to select.

; RUN: split-file %s %t

; RUN: not llc -O0 -mtriple=spirv64-- %t/load-vector.ll -o /dev/null 2>&1 | FileCheck --check-prefix=FAIL-LOAD-VEC %s

; RUN: not llc -O0 -mtriple=spirv64-- %t/load-volatile.ll -o /dev/null 2>&1 | FileCheck --check-prefix=FAIL-LOAD-VOL %s

; RUN: not llc -O0 -mtriple=spirv64-- %t/store-vector.ll -o /dev/null 2>&1 | FileCheck --check-prefix=FAIL-STORE-VEC %s

; RUN: not llc -O0 -mtriple=spirv64-- %t/store-volatile.ll -o /dev/null 2>&1 | FileCheck --check-prefix=FAIL-STORE-VOL %s

; FAIL-LOAD-VEC: error:{{.*}}atomic load is only allowed for integer or floating point types
; FAIL-LOAD-VOL: error:{{.*}}atomic load of volatile memory is not supported
; FAIL-STORE-VEC: error:{{.*}}atomic store is only allowed for integer or floating point types
; FAIL-STORE-VOL: error:{{.*}}atomic store of volatile memory is not supported

;--- load-vector.ll
define <2 x i32> @load_vector_acquire(ptr addrspace(1) %ptr) {
  %val = load atomic <2 x i32>, ptr addrspace(1) %ptr acquire, align 8
  ret <2 x i32> %val
}

;--- load-volatile.ll
define i32 @load_i32_acquire_device_volatile(ptr addrspace(1) %ptr) {
  %val = load atomic volatile i32, ptr addrspace(1) %ptr syncscope("device") acquire, align 4
  ret i32 %val
}

;--- store-vector.ll
define void @store_vector_release(ptr addrspace(1) %ptr, <2 x i32> %val) {
  store atomic <2 x i32> %val, ptr addrspace(1) %ptr release, align 8
  ret void
}

;--- store-volatile.ll
define void @store_i32_release_device_volatile(ptr addrspace(1) %ptr, i32 %val) {
  store atomic volatile i32 %val, ptr addrspace(1) %ptr syncscope("device") release, align 4
  ret void
}
