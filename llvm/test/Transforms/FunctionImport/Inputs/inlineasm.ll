@myvar = internal constant i8 1, align 1
@llvm.used = appending global [1 x i8*] [i8* @myvar], section "llvm.metadata"

define void @foo(i64* %v) #0 {
entry:
  %v.addr = alloca i64*, align 8
  store i64* %v, i64** %v.addr, align 8
  %0 = load i64*, i64** %v.addr, align 8
  call void asm sideeffect "movzbl     myvar(%rip), %eax\0A\09movq %rax, $0\0A\09", "=*m,~{eax},~{dirflag},~{fpsr},~{flags}"(i64* elementtype(i64) %0) #1
  ret void
}
