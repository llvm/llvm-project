; A barrier that AIRTGBarrierInsert adds takes the form of the module's
; existing air.wg.barrier: (flags, order, scope) here, the AIR 2.9 form. It
; used to add a two-operand call to the three-operand declaration, a call
; whose own function type the writer had not registered, which produced
; bitcode no reader accepts ("Explicit call type is not a function type").
; RUN: llc -mtriple=air %s -o - | FileCheck %s
; RUN: llc -mtriple=air -filetype=obj -air-output=air %s -o %t.air
; RUN: llvm-dis %t.air -o - | FileCheck %s --check-prefix=BC

; CHECK-LABEL: define void @kernel
; CHECK-NOT: call void @air.wg.barrier(i32 2, i32 1)
; CHECK: call void @air.wg.barrier(i32 2, i32 5, i32 1)
; CHECK-NOT: call void @air.wg.barrier(i32 2, i32 1)
; BC: call void @air.wg.barrier(i32 2, i32 5, i32 1)

@tg = internal addrspace(3) global [32 x i32] undef, align 4

declare { i32, i32, i32 } @air.thread_position_in_threadgroup.v3i32()
declare void @air.wg.barrier(i32, i32, i32)

define void @kernel(ptr addrspace(1) %in, ptr addrspace(1) %out) {
entry:
  %lt = call { i32, i32, i32 } @air.thread_position_in_threadgroup.v3i32()
  %lid = extractvalue { i32, i32, i32 } %lt, 0
  %ip = getelementptr i32, ptr addrspace(1) %in, i32 %lid
  %v = load i32, ptr addrspace(1) %ip
  %sp = getelementptr [32 x i32], ptr addrspace(3) @tg, i32 0, i32 %lid
  store i32 %v, ptr addrspace(3) %sp
  %rl = sub i32 31, %lid
  %rp = getelementptr [32 x i32], ptr addrspace(3) @tg, i32 0, i32 %rl
  %r = load i32, ptr addrspace(3) %rp
  %op = getelementptr i32, ptr addrspace(1) %out, i32 %lid
  store i32 %r, ptr addrspace(1) %op
  call void @air.wg.barrier(i32 2, i32 5, i32 1)
  ret void
}
