; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -agx-max-unskipped-instrs=0 -verify-machineinstrs < %s | FileCheck %s --check-prefix=ALWAYS

; A short ALU-only body runs with an empty execution mask instead of being
; jumped over (B31); a body with a memory access, or any body with the
; threshold at 0, is jumped over.

declare i32 @llvm.agx.thread.position.in.grid.x()

; CHECK-LABEL: {{^}}alu:
; CHECK:         if_icmp
; CHECK-NOT:     jmp_exec_none
; CHECK:         pop_exec
; ALWAYS-LABEL: {{^}}alu:
; ALWAYS:        if_icmp
; ALWAYS-NEXT:   jmp_exec_none
define agx_kernel void @alu(ptr addrspace(1) %out, i32 %n) {
entry:
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %c = icmp ult i32 %t, %n
  br i1 %c, label %then, label %join
then:
  %x = mul i32 %t, %t
  br label %join
join:
  %r = phi i32 [ %x, %then ], [ %t, %entry ]
  %p = getelementptr i32, ptr addrspace(1) %out, i32 %t
  store i32 %r, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: {{^}}memory:
; CHECK:         if_icmp
; CHECK-NEXT:    jmp_exec_none
define agx_kernel void @memory(ptr addrspace(1) %out, i32 %n) {
entry:
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %c = icmp ult i32 %t, %n
  br i1 %c, label %then, label %join
then:
  %p = getelementptr i32, ptr addrspace(1) %out, i32 %t
  store i32 %t, ptr addrspace(1) %p
  br label %join
join:
  ret void
}
