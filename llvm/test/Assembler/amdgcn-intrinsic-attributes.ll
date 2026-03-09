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

; CHECK: declare void @llvm.amdgcn.s.wait.event(i16 immarg) #1
define void @s_wait_event() {
  call void @llvm.amdgcn.s.wait.event(i16 0)
  ret void
}

; CHECK: declare void @llvm.amdgcn.s.wait.event.export.ready() #1
define void @s_wait_event_export_ready() {
  call void @llvm.amdgcn.s.wait.event.export.ready()
  ret void
}

; Test assumed range
; CHECK: declare noundef range(i32 32, 65) i32 @llvm.amdgcn.wavefrontsize() #2
define i32 @wavefrontsize() {
  %ret = call i32 @llvm.amdgcn.wavefrontsize()
  ret i32 %ret
}

; CHECK: attributes #0 = { convergent nocallback nofree nounwind willreturn memory(argmem: readwrite) }
; CHECK: attributes #1 = { nocallback nofree nounwind willreturn }
; CHNCK: attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
