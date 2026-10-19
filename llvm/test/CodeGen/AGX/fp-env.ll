; RUN: llc -mtriple=agx -mcpu=agx-g13g -o /dev/null -print-after=agx-lower-fp-env < %s 2>&1 | FileCheck %s

; Round-to-nearest constrained operations become ordinary ones.
; CHECK-LABEL: @nearest(
; CHECK: %r = fadd float %a, %b
define void @nearest(ptr addrspace(1) %out, float %a, float %b) #0 {
  %r = call float @llvm.experimental.constrained.fadd.f32(float %a, float %b, metadata !"round.tonearest", metadata !"fpexcept.ignore") #0
  store float %r, ptr addrspace(1) %out
  ret void
}

; The mode register starts at round to nearest, so a dynamic rounding with
; no set.fpmode is an ordinary operation too.
; CHECK-LABEL: @dynamic_default(
; CHECK: %r = fmul float %a, %b
define void @dynamic_default(ptr addrspace(1) %out, float %a, float %b) #0 {
  %r = call float @llvm.experimental.constrained.fmul.f32(float %a, float %b, metadata !"round.dynamic", metadata !"fpexcept.ignore") #0
  store float %r, ptr addrspace(1) %out
  ret void
}

; A known directed mode is computed in software on the bits, and
; get.fpmode reads back the value set.
; CHECK-LABEL: @toward_zero(
; CHECK-NOT: call {{.*}}constrained
; CHECK-NOT: fadd float %a, %b{{$}}
; CHECK: bitcast float %a to i32
; CHECK-NOT: llvm.agx.{{[sg]}}et.fpmode
; CHECK: store i32 0, ptr addrspace(1) %mode
define void @toward_zero(ptr addrspace(1) %out, ptr addrspace(1) %mode, float %a, float %b) #0 {
  call void @llvm.agx.set.fpmode(i32 0)
  %r = call float @llvm.experimental.constrained.fadd.f32(float %a, float %b, metadata !"round.dynamic", metadata !"fpexcept.ignore") #0
  store float %r, ptr addrspace(1) %out
  %m = call i32 @llvm.agx.get.fpmode()
  store i32 %m, ptr addrspace(1) %mode
  ret void
}

; A mode only known at run time is read from its value: the f32 rounding
; field (bits 0-1) and the f32 flush bit (bit 4).
; CHECK-LABEL: @unknown_mode(
; CHECK-NOT: llvm.agx.{{[sg]}}et.fpmode
; CHECK-DAG: and i32 %rm, 3
; CHECK-DAG: and i32 %rm, 16
define void @unknown_mode(ptr addrspace(1) %out, i32 %rm, float %a, float %b) #0 {
  call void @llvm.agx.set.fpmode(i32 %rm)
  %r = call float @llvm.experimental.constrained.fadd.f32(float %a, float %b, metadata !"round.dynamic", metadata !"fpexcept.ignore") #0
  store float %r, ptr addrspace(1) %out
  ret void
}

declare float @llvm.experimental.constrained.fadd.f32(float, float, metadata, metadata)
declare float @llvm.experimental.constrained.fmul.f32(float, float, metadata, metadata)
declare void @llvm.agx.set.fpmode(i32)
declare i32 @llvm.agx.get.fpmode()

attributes #0 = { strictfp }
