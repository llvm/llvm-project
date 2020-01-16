; RUN: not llc -global-isel -march=amdgcn -mcpu=tonga < %S/../lds-zero-initializer.ll 2>&1 | FileCheck %s

; FIXME: Select should succeed
; CHECK: error: <unknown>:0:0: in function load_zeroinit_lds_global void (i32 addrspace(1)*, i1): unsupported initializer for address space
; CHECK: LLVM ERROR: cannot select: %16:sreg_32(p3) = G_GLOBAL_VALUE @lds (in function: load_zeroinit_lds_global)
