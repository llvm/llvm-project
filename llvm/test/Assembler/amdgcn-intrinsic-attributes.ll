; REQUIRES: amdgpu-registered-target

; RUN: llvm-as < %s | llvm-dis | FileCheck %s

; Test assumed alignment parameter

; CHECK: declare i32 @llvm.amdgcn.ds.append.p3(ptr addrspace(3) align 4 captures(none), i1 immarg) #0

define i32 @ds_append(ptr addrspace(3) %ptr) {
  %ret = call i32 @llvm.amdgcn.ds.append.p3(ptr addrspace(3) %ptr, i1 false)
  ret i32 %ret
}

; Test assumed alignment parameter
; CHECK: declare i32 @llvm.amdgcn.ds.consume.p3(ptr addrspace(3) align 4 captures(none), i1 immarg) #0
define i32 @ds_consume(ptr addrspace(3) %ptr) {
  %ret = call i32 @llvm.amdgcn.ds.consume.p3(ptr addrspace(3) %ptr, i1 false)
  ret i32 %ret
}

; CHECK: attributes #0 = { convergent nocallback nofree nounwind willreturn memory(argmem: readwrite) }
