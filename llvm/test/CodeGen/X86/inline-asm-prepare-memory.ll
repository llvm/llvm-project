; RUN: llc -O0 -mtriple=x86_64-unknown-linux-gnu -stop-after=inline-asm-prepare < %s | FileCheck %s

define void @test1(i32 %x) {
; CHECK-LABEL: @test1
; CHECK:         %asm_mem = alloca i32
; CHECK-NEXT:    store i32 %x, ptr %asm_mem
; CHECK-NEXT:    call i32 asm sideeffect "mov $1, $0", "=r,*rm,~{dirflag},~{fpsr},~{flags}"(ptr %asm_mem)
entry:
  %0 = call i32 asm sideeffect "mov $1, $0", "=r,rm,~{dirflag},~{fpsr},~{flags}"(i32 %x)
  ret void
}

define void @test2(ptr %p) {
; CHECK-LABEL: @test2
; CHECK:         %asm_mem = alloca i32
; CHECK-NEXT:    call void asm sideeffect "mov $1, $0", "=*rm,~{dirflag},~{fpsr},~{flags}"(ptr elementtype(i32) %asm_mem)
; CHECK-NEXT:    %[[VAL1:.*]] = load i32, ptr %asm_mem
; CHECK-NEXT:    store i32 %[[VAL1]], ptr %p
entry:
  %0 = call i32 asm sideeffect "mov $1, $0", "=rm,~{dirflag},~{fpsr},~{flags}"()
  store i32 %0, ptr %p
  ret void
}

define void @test3(ptr %x_ptr) {
; CHECK-LABEL: @test3
; CHECK:         %asm_mem = alloca i32
; CHECK-NEXT:    %x = load i32, ptr %x_ptr
; CHECK-NEXT:    store i32 %x, ptr %asm_mem
; CHECK-NEXT:    call void asm sideeffect "inc $0", "=*rm,0,~{dirflag},~{fpsr},~{flags}"(ptr elementtype(i32) %asm_mem, ptr %asm_mem)
; CHECK-NEXT:    %[[VAL2:.*]] = load i32, ptr %asm_mem
; CHECK-NEXT:    store i32 %[[VAL2]], ptr %x_ptr
entry:
  %x = load i32, ptr %x_ptr
  %0 = call i32 asm sideeffect "inc $0", "=rm,0,~{dirflag},~{fpsr},~{flags}"(i32 %x)
  store i32 %0, ptr %x_ptr
  ret void
}
