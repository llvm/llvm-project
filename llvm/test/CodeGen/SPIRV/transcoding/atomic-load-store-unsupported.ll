; Verify that atomic load/store of vectors correctly fail to select.

; RUN: split-file %s %t

; RUN: not llc -O0 -mtriple=spirv64-- %t/load-vector.ll -o /dev/null 2>&1 | FileCheck --check-prefix=FAIL-LOAD-VEC %s

; RUN: not llc -O0 -mtriple=spirv64-- %t/store-vector.ll -o /dev/null 2>&1 | FileCheck --check-prefix=FAIL-STORE-VEC %s

; FAIL-LOAD-VEC: error:{{.*}}atomic load is only allowed for integer or floating point types
; FAIL-STORE-VEC: error:{{.*}}atomic store is only allowed for integer or floating point types

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
