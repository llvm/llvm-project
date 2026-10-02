; RUN: opt < %s -S -passes=openmp-opt | FileCheck %s

; Check that the call to __cxx_global_var_init is not incorrectly optimized out.

target triple = "spirv64-intel"

%struct.S = type { i32 }

@s = addrspace(1) global %struct.S zeroinitializer
@llvm.global_ctors = appending global [1 x { i32, ptr addrspace(9), ptr addrspace(9) }] [{ i32, ptr addrspace(9), ptr addrspace(9) } { i32 65535, ptr addrspace(9) @_GLOBAL__sub_I_ctor_dtor.cpp, ptr addrspace(9) null }]

define spir_func void @__cxx_global_var_init() addrspace(9) {
entry:
  call spir_func addrspace(9) void @_ZN1SC2Ev(ptr addrspace(4) addrspacecast (ptr addrspace(1) @s to ptr addrspace(4)))
  ret void
}

define spir_func void @_ZN1SC2Ev(ptr addrspace(4) %this) addrspace(9) {
entry:
  %this.addr = alloca ptr addrspace(4)
  %this.addr.ascast = addrspacecast ptr %this.addr to ptr addrspace(4)
  store ptr addrspace(4) %this, ptr addrspace(4) %this.addr.ascast
  %this1 = load ptr addrspace(4), ptr addrspace(4) %this.addr.ascast
  %i = getelementptr %struct.S, ptr addrspace(4) %this1, i32 0, i32 0
  store i32 7, ptr addrspace(4) %i
  ret void
}

; CHECK-LABEL: define internal spir_func void @_GLOBAL__sub_I_ctor_dtor.cpp()
; CHECK-NEXT: entry:
; CHECK-NEXT: call spir_func addrspace(9) void @__cxx_global_var_init{{.*}}()
; CHECK-NEXT: ret void

define internal spir_func void @_GLOBAL__sub_I_ctor_dtor.cpp() addrspace(9) {
entry:
  call spir_func addrspace(9) void @__cxx_global_var_init()
  ret void
}

!llvm.module.flags = !{!2, !3}

!2 = !{i32 7, !"openmp", i32 51}
!3 = !{i32 7, !"openmp-device", i32 51}
