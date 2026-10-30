; Metadata passed as a call argument is written as its metadata ID: a string
; (the rounding of a constrained intrinsic) and a node keep their identity
; instead of reading back as another string of the module (here the module
; flag's).
; RUN: llc -mtriple=air -filetype=obj -air-output=air %s -o %t.air
; RUN: llvm-dis %t.air -o - | FileCheck %s

; CHECK-LABEL: define {{.*}}@k(
; CHECK: call float @llvm.experimental.constrained.fadd.f32(float %{{[0-9]+}}, float %{{[0-9]+}}, metadata !"round.towardzero", metadata !"fpexcept.ignore")
; CHECK-LABEL: define {{.*}}@k2(
; CHECK: call i32 @llvm.read_register.i32(metadata ![[N:[0-9]+]])
; CHECK: ![[N]] = !{!"node-arg"}

target triple = "air"

declare float @llvm.experimental.constrained.fadd.f32(float, float, metadata, metadata)
declare i32 @llvm.read_register.i32(metadata)

define void @k(ptr addrspace(1) %p) "air-kernel" {
  %x = load float, ptr addrspace(1) %p
  %y = call float @llvm.experimental.constrained.fadd.f32(float %x, float %x, metadata !"round.towardzero", metadata !"fpexcept.ignore") strictfp
  store float %y, ptr addrspace(1) %p
  ret void
}

define void @k2(ptr addrspace(1) %p) "air-kernel" {
  %r = call i32 @llvm.read_register.i32(metadata !1)
  store i32 %r, ptr addrspace(1) %p
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 2, !"agx-stack-size", i32 1024}
!1 = !{!"node-arg"}
