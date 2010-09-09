// RUN: %llvmgcc %s -S -o - | FileCheck %s
// Exercise various use cases for local asm "register variables".
// XFAIL: *
// XTARGET: x86_64,i686,i386

int foo() {
// CHECK: %a = alloca i32

  register int a asm("rsi")=5;
// CHECK: store i32 5, i32* %a, align 4

  asm volatile("; %0 This asm defines rsi" : "=r"(a));
// CHECK: %asmtmp = call i32 asm sideeffect "; $0 This asm defines rsi", "={rsi}
// CHECK: store i32 %asmtmp, i32* %a

  a = 42;
// CHECK:  store i32 42, i32* %a, align 4

  asm volatile("; %0 This asm uses rsi" : : "r"(a));
// CHECK:  %1 = load i32* %a, align 4                    
// CHECK:  call void asm sideeffect "", "{rsi}"(i32 %1) nounwind
// CHECK:  %2 = call i32 asm sideeffect "", "={rsi}"() nounwind
// CHECK:  call void asm sideeffect "; $0 This asm uses rsi", "{rsi},~{dirflag},~{fpsr},~{flags}"(i32 %2)

  return a;
// CHECK:  %3 = load i32* %a, align 4
// CHECK:  call void asm sideeffect "", "{rsi}"(i32 %3) nounwind
// CHECK:  %4 = call i32 asm sideeffect "", "={rsi}"() nounwind 
// CHECK:  store i32 %4, i32* %0, align 4
// CHECK:  %5 = load i32* %0, align 4                     
// CHECK:  store i32 %5, i32* %retval, align 4
}
