; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu | FileCheck %s -check-prefix=X64
; RUN: llc < %s -mtriple=i686-unknown-linux-gnu | FileCheck %s -check-prefix=X32

; CHECK-LABEL: foo

; Check the functionality of the local stack symbol table ordering
; heuristics.
; The test has a bunch of locals of various sizes that are referenced a
; different number of times.
;
; a   : 120B, 9 uses,   density = 0.075
; aa  : 4000B, 1 use,   density = 0.00025
; b   : 4B, 1 use,      density = 0.25
; cc  : 4000B, 2 uses   density = 0.0005
; d   : 4B, 2 uses      density = 0.5
; e   : 4B, 3 uses      density = 0.75
; f   : 4B, 4 uses      density = 1
;
; Given the size, number of uses and calculated density (uses / size), we're
; going to hope that f gets allocated closest to the stack pointer,
; followed by e, d, b, then a (to check for just a few).
; We use gnu-inline asm between calls to prevent registerization of addresses
; so that we get exact counts.
;
; The test is taken from something like this:
; void foo()
; {
;   int f; // 4 uses.          4 / 4 = 1
;   int a[30]; // 9 uses.      8 / 120 = 0.06
;   int aa[1000]; // 1 use.    1 / 4000 =
;   int e; // 3 uses.          3 / 4 = 0.75
;   int cc[1000]; // 2 uses.   2 / 4000 = 
;   int b; // 1 use.           1 / 4 = 0.25
;   int d; // 2 uses.          2 / 4 = 0.5
;   int aaa[1000]; // 2 uses.  2 / 4000
;
; 
;   check_a(&a);
;   bar1(&aaa);
;   asm ("":::"esi","edi","ebp","ebx","rbx","r12","r13","r14","r15","rbp");
;   bar1(&a);
;   check_f(&f);
;   asm ("":::"esi","edi","ebp","ebx","rbx","r12","r13","r14","r15","rbp");
;   bar1(&a);
;   bar3(&aa, &aaa, &cc);
;   asm ("":::"esi","edi","ebp","ebx","rbx","r12","r13","r14","r15","rbp");
;   bar2(&a,&cc);
;   check_b(&b);
;   asm ("":::"esi","edi","ebp","ebx","rbx","r12","r13","r14","r15","rbp");
;   bar1(&a);
;   asm ("":::"esi","edi","ebp","ebx","rbx","r12","r13","r14","r15","rbp");
;   bar2(&a, &f);
;   check_e(&e);
;   asm ("":::"esi","edi","ebp","ebx","rbx","r12","r13","r14","r15","rbp");
;   bar1(&a);
;   asm ("":::"esi","edi","ebp","ebx","rbx","r12","r13","r14","r15","rbp");
;   bar2(&e, &f);
;   check_d(&d);
;   bar1(&a);
;   asm ("":::"esi","edi","ebp","ebx","rbx","r12","r13","r14","r15","rbp");
;   bar3(&d, &e, &f);
;   asm ("":::"esi","edi","ebp","ebx","rbx","r12","r13","r14","r15","rbp");
;   bar1(&a);
; }
;
; X64: leaq 16(%rsp), %rdi
; X64: callq check_a
; X64: callq bar1
; X64: callq bar1
; X64: movq %rsp, %rdi
; X64: callq check_f
; X64: callq bar1
; X64: callq bar3
; X64: callq bar2
; X64: leaq 12(%rsp), %rdi
; X64: callq check_b
; X64: callq bar1
; X64: callq bar2
; X64: leaq 4(%rsp), %rdi
; X64: callq check_e
; X64: callq bar1
; X64: callq bar2
; X64: leaq 8(%rsp), %rdi
; X64: callq check_d

; X32: leal 32(%esp)
; X32: calll check_a
; X32: calll bar1
; X32: calll bar1
; X32: leal 16(%esp)
; X32: calll check_f
; X32: calll bar1
; X32: calll bar3
; X32: calll bar2
; X32: leal 28(%esp)
; X32: calll check_b
; X32: calll bar1
; X32: calll bar2
; X32: leal 20(%esp)
; X32: calll check_e
; X32: calll bar1
; X32: calll bar2
; X32: leal 24(%esp)
; X32: calll check_d


define void @foo() nounwind uwtable {
entry:
  %f = alloca i32, align 4
  %a = alloca [30 x i32], align 16
  %aa = alloca [1000 x i32], align 16
  %e = alloca i32, align 4
  %cc = alloca [1000 x i32], align 16
  %b = alloca i32, align 4
  %d = alloca i32, align 4
  %aaa = alloca [1000 x i32], align 16
  call void @llvm.lifetime.start.p0(i64 4, ptr %f) #1
  call void @llvm.lifetime.start.p0(i64 120, ptr %a) #1
  call void @llvm.lifetime.start.p0(i64 4000, ptr %aa) #1
  call void @llvm.lifetime.start.p0(i64 4, ptr %e) #1
  call void @llvm.lifetime.start.p0(i64 4000, ptr %cc) #1
  call void @llvm.lifetime.start.p0(i64 4, ptr %b) #1
  call void @llvm.lifetime.start.p0(i64 4, ptr %d) #1
  call void @llvm.lifetime.start.p0(i64 4000, ptr %aaa) #1
  %call = call i32 (ptr, ...) @check_a(ptr %a)
  %call1 = call i32 (ptr, ...) @bar1(ptr %aaa)
  call void asm sideeffect "", "~{esi},~{edi},~{ebp},~{ebx},~{rbx},~{r12},~{r13},~{r14},~{r15},~{rbp},~{dirflag},~{fpsr},~{flags}"() #1
  %call2 = call i32 (ptr, ...) @bar1(ptr %a)
  %call3 = call i32 (ptr, ...) @check_f(ptr %f)
  call void asm sideeffect "", "~{esi},~{edi},~{ebp},~{ebx},~{rbx},~{r12},~{r13},~{r14},~{r15},~{rbp},~{dirflag},~{fpsr},~{flags}"() #1
  %call4 = call i32 (ptr, ...) @bar1(ptr %a)
  %call5 = call i32 (ptr, ptr, ptr, ...) @bar3(ptr %aa, ptr %aaa, ptr %cc)
  call void asm sideeffect "", "~{esi},~{edi},~{ebp},~{ebx},~{rbx},~{r12},~{r13},~{r14},~{r15},~{rbp},~{dirflag},~{fpsr},~{flags}"() #1
  %call6 = call i32 (ptr, ptr, ...) @bar2(ptr %a, ptr %cc)
  %call7 = call i32 (ptr, ...) @check_b(ptr %b)
  call void asm sideeffect "", "~{esi},~{edi},~{ebp},~{ebx},~{rbx},~{r12},~{r13},~{r14},~{r15},~{rbp},~{dirflag},~{fpsr},~{flags}"() #1
  %call8 = call i32 (ptr, ...) @bar1(ptr %a)
  call void asm sideeffect "", "~{esi},~{edi},~{ebp},~{ebx},~{rbx},~{r12},~{r13},~{r14},~{r15},~{rbp},~{dirflag},~{fpsr},~{flags}"() #1
  %call9 = call i32 (ptr, ptr, ...) @bar2(ptr %a, ptr %f)
  %call10 = call i32 (ptr, ...) @check_e(ptr %e)
  call void asm sideeffect "", "~{esi},~{edi},~{ebp},~{ebx},~{rbx},~{r12},~{r13},~{r14},~{r15},~{rbp},~{dirflag},~{fpsr},~{flags}"() #1
  %call11 = call i32 (ptr, ...) @bar1(ptr %a)
  call void asm sideeffect "", "~{esi},~{edi},~{ebp},~{ebx},~{rbx},~{r12},~{r13},~{r14},~{r15},~{rbp},~{dirflag},~{fpsr},~{flags}"() #1
  %call12 = call i32 (ptr, ptr, ...) @bar2(ptr %e, ptr %f)
  %call13 = call i32 (ptr, ...) @check_d(ptr %d)
  %call14 = call i32 (ptr, ...) @bar1(ptr %a)
  call void asm sideeffect "", "~{esi},~{edi},~{ebp},~{ebx},~{rbx},~{r12},~{r13},~{r14},~{r15},~{rbp},~{dirflag},~{fpsr},~{flags}"() #1
  %call15 = call i32 (ptr, ptr, ptr, ...) @bar3(ptr %d, ptr %e, ptr %f)
  call void asm sideeffect "", "~{esi},~{edi},~{ebp},~{ebx},~{rbx},~{r12},~{r13},~{r14},~{r15},~{rbp},~{dirflag},~{fpsr},~{flags}"() #1
  %call16 = call i32 (ptr, ...) @bar1(ptr %a)
  call void @llvm.lifetime.end.p0(i64 4000, ptr %aaa) #1
  call void @llvm.lifetime.end.p0(i64 4, ptr %d) #1
  call void @llvm.lifetime.end.p0(i64 4, ptr %b) #1
  call void @llvm.lifetime.end.p0(i64 4000, ptr %cc) #1
  call void @llvm.lifetime.end.p0(i64 4, ptr %e) #1
  call void @llvm.lifetime.end.p0(i64 4000, ptr %aa) #1
  call void @llvm.lifetime.end.p0(i64 120, ptr %a) #1
  call void @llvm.lifetime.end.p0(i64 4, ptr %f) #1
  ret void
}

; Function Attrs: nounwind
declare void @llvm.lifetime.start.p0(i64, ptr nocapture) #1

declare i32 @check_a(...) #2
declare i32 @bar1(...) #2
declare i32 @check_f(...) #2
declare i32 @bar3(...) #2
declare i32 @bar2(...) #2
declare i32 @check_b(...) #2
declare i32 @check_e(...) #2
declare i32 @check_d(...) #2

; Function Attrs: nounwind
declare void @llvm.lifetime.end.p0(i64, ptr nocapture) #1

