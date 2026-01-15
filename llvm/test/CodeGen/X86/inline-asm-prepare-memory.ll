; RUN: llc -O0 -mtriple=x86_64-unknown-linux-gnu -stop-after=inline-asm-prepare < %s | FileCheck %s

define void @func_rm_input(i32 %x) {
; CHECK-LABEL: @func_rm_input
; CHECK: %asm_mem = alloca i32
; CHECK: store i32 %x, ptr %asm_mem
; CHECK: call i32 asm sideeffect "mov $1, $0", "=r,m,~{dirflag},~{fpsr},~{flags}"(ptr %asm_mem)
entry:
  %0 = call i32 asm sideeffect "mov $1, $0", "=r,rm,~{dirflag},~{fpsr},~{flags}"(i32 %x)
  ret void
}

define void @func_rm_output(ptr %p) {
; CHECK-LABEL: @func_rm_output
; CHECK: %asm_mem = alloca i32
; CHECK: call void asm sideeffect "mov $1, $0", "=*m,~{dirflag},~{fpsr},~{flags}"(ptr elementtype(i32) %asm_mem)
; CHECK: %[[VAL:.*]] = load i32, ptr %asm_mem
; CHECK: store i32 %[[VAL]], ptr %p
entry:
  %0 = call i32 asm sideeffect "mov $1, $0", "=rm,~{dirflag},~{fpsr},~{flags}"()
  store i32 %0, ptr %p
  ret void
}

define void @func_rm_inout(ptr %x_ptr) {
; CHECK-LABEL: @func_rm_inout
; CHECK: %x = load i32, ptr %x_ptr
; CHECK: %asm_mem = alloca i32
; CHECK: store i32 %x, ptr %asm_mem
; CHECK: call void asm sideeffect "inc $0", "=*m,0,~{dirflag},~{fpsr},~{flags}"(ptr elementtype(i32) %asm_mem, ptr %asm_mem)
; CHECK: %[[VAL2:.*]] = load i32, ptr %asm_mem
; CHECK: store i32 %[[VAL2]], ptr %x_ptr
entry:
  %x = load i32, ptr %x_ptr
  %0 = call i32 asm sideeffect "inc $0", "=rm,0,~{dirflag},~{fpsr},~{flags}"(i32 %x)
  store i32 %0, ptr %x_ptr
  ret void
}
