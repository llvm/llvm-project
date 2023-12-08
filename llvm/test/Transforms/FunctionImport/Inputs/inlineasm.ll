@myvar = internal constant i8 1, align 1
@llvm.used = appending global [1 x ptr] [ptr @myvar], section "llvm.metadata"

define void @foo(ptr %v) #0 {
entry:
  %v.addr = alloca ptr, align 8
  store ptr %v, ptr %v.addr, align 8
  %0 = load ptr, ptr %v.addr, align 8
  call void asm sideeffect "movzbl     myvar(%rip), %eax\0A\09movq %rax, $0\0A\09", "=*m,~{eax},~{dirflag},~{fpsr},~{flags}"(ptr elementtype(i64) %0) #1
  ret void
}
