; RUN: llc < %s | FileCheck %s

define void @test1() {
; CHECK-LABEL: test1:
; CHECK: vmovaps %xmm0, %xmm0
; CHECK: vmovaps %ymm0, %ymm0
; CHECK: vmovaps %zmm0, %zmm0
  tail call void asm sideeffect "vmovaps ${0:x}, ${0:x}", "{xmm0},~{dirflag},~{fpsr},~{flags}"(i32 0)
  tail call void asm sideeffect "vmovaps ${0:t}, ${0:t}", "{xmm0},~{dirflag},~{fpsr},~{flags}"(i32 0)
  tail call void asm sideeffect "vmovaps ${0:g}, ${0:g}", "{xmm0},~{dirflag},~{fpsr},~{flags}"(i32 0)
  ret void
}

define void @test2() {
; CHECK-LABEL: test2:
; CHECK: vmovaps %xmm0, %xmm0
; CHECK: vmovaps %ymm0, %ymm0
; CHECK: vmovaps %zmm0, %zmm0
  tail call void asm sideeffect inteldialect "vmovaps ${0:x}, ${0:x}", "{xmm0},~{dirflag},~{fpsr},~{flags}"(i32 0)
  tail call void asm sideeffect inteldialect "vmovaps ${0:t}, ${0:t}", "{xmm0},~{dirflag},~{fpsr},~{flags}"(i32 0)
  tail call void asm sideeffect inteldialect "vmovaps ${0:g}, ${0:g}", "{xmm0},~{dirflag},~{fpsr},~{flags}"(i32 0)
  ret void
}

define void @test3() {
; CHECK-LABEL: test3:
; CHECK: movb %al, %al
; CHECK: movb %ah, %ah
; CHECK: movw %ax, %ax
; CHECK: movl %eax, %eax
; CHECK: movq %rax, %rax
  tail call void asm sideeffect "mov ${0:b}, ${0:b}", "{eax},~{dirflag},~{fpsr},~{flags}"(i32 0)
  tail call void asm sideeffect "mov ${0:h}, ${0:h}", "{eax},~{dirflag},~{fpsr},~{flags}"(i32 0)
  tail call void asm sideeffect "mov ${0:w}, ${0:w}", "{eax},~{dirflag},~{fpsr},~{flags}"(i32 0)
  tail call void asm sideeffect "mov ${0:k}, ${0:k}", "{eax},~{dirflag},~{fpsr},~{flags}"(i32 0)
  tail call void asm sideeffect "mov ${0:q}, ${0:q}", "{eax},~{dirflag},~{fpsr},~{flags}"(i32 0)
  ret void
}

define void @test4() {
; CHECK-LABEL: test4:
; CHECK: movb %al, %al
; CHECK: movb %ah, %ah
; CHECK: movw %ax, %ax
; CHECK: movl %eax, %eax
; CHECK: movq %rax, %rax
  tail call void asm sideeffect inteldialect "mov ${0:b}, ${0:b}", "{eax},~{dirflag},~{fpsr},~{flags}"(i32 0)
  tail call void asm sideeffect inteldialect "mov ${0:h}, ${0:h}", "{eax},~{dirflag},~{fpsr},~{flags}"(i32 0)
  tail call void asm sideeffect inteldialect "mov ${0:w}, ${0:w}", "{eax},~{dirflag},~{fpsr},~{flags}"(i32 0)
  tail call void asm sideeffect inteldialect "mov ${0:k}, ${0:k}", "{eax},~{dirflag},~{fpsr},~{flags}"(i32 0)
  tail call void asm sideeffect inteldialect "mov ${0:q}, ${0:q}", "{eax},~{dirflag},~{fpsr},~{flags}"(i32 0)
  ret void
}
