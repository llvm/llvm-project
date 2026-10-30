; Metal's typed-pointer reader rejects a library ("Unexpected bitcode file!")
; whose GEP source type differs from its base pointer's pointee, or whose
; air.atomic call takes a pointer to another type.
; RUN: llc -mtriple=air -filetype=obj -air-output=air %s -o %t.air
; RUN: llvm-dis %t.air -o - | FileCheck %s

; A float stored through an i32 GEP of a float buffer: the writer casts the
; buffer to i32* for the GEP, which is written as an i32 GEP (not float).
; CHECK-LABEL: define void @pun_store(
; CHECK: [[B:%.*]] = bitcast ptr addrspace(1) %1 to ptr addrspace(1)
; CHECK: [[G:%.*]] = getelementptr i32, ptr addrspace(1) [[B]],
; CHECK: [[S:%.*]] = bitcast ptr addrspace(1) [[G]] to ptr addrspace(1)
; CHECK: store float 1.000000e+00, ptr addrspace(1) [[S]]
define void @pun_store(ptr addrspace(1) %in, ptr addrspace(1) %out, i32 %i) "air-kernel" {
  %po = getelementptr i32, ptr addrspace(1) %out, i32 %i
  store float 1.0, ptr addrspace(1) %po
  ret void
}

; An i32 compare-exchange on a float threadgroup global (an atomic float add
; as a loop) goes through an i32* cast of the global.
@tg = internal addrspace(3) global float undef, align 4

; CHECK-LABEL: define void @tg_cmpxchg(
; CHECK: [[C:%.*]] = bitcast ptr addrspace(3) @tg to ptr addrspace(3)
; CHECK: call i32 @air.atomic.local.cmpxchg.weak.i32(ptr addrspace(3) [[C]],
define void @tg_cmpxchg(ptr addrspace(1) %out, i32 %v) "air-kernel" {
  store float 0.0, ptr addrspace(3) @tg
  %x = cmpxchg ptr addrspace(3) @tg, i32 0, i32 %v monotonic monotonic
  %o = extractvalue { i32, i1 } %x, 0
  store i32 %o, ptr addrspace(1) %out
  ret void
}
