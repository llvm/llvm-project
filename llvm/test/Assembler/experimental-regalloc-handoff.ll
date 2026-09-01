; RUN: llvm-as %s -o - | llvm-dis -o - | FileCheck %s

define i32 @identity_i32(i32 %x) {
  %y = call i32 @llvm.experimental.regalloc.handoff(i32 %x, metadata !0)
  ret i32 %y
}

declare i32 @llvm.experimental.regalloc.handoff(i32, metadata)

; CHECK: declare i32 @llvm.experimental.regalloc.handoff(i32, metadata) #[[ATTR:[0-9]+]]
; CHECK: {{^}}attributes #[[ATTR]] = { nocallback nocreateundeforpoison nofree nomerge nosync nounwind willreturn memory(inaccessiblemem: readwrite) }{{$}}

!0 = !{!"amdgpu.vgpr"}
