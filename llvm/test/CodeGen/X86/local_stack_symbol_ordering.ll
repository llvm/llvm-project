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
  %0 = bitcast i32* %f to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %0) #1
  %1 = bitcast [30 x i32]* %a to i8*
  call void @llvm.lifetime.start.p0i8(i64 120, i8* %1) #1
  %2 = bitcast [1000 x i32]* %aa to i8*
  call void @llvm.lifetime.start.p0i8(i64 4000, i8* %2) #1
  %3 = bitcast i32* %e to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %3) #1
  %4 = bitcast [1000 x i32]* %cc to i8*
  call void @llvm.lifetime.start.p0i8(i64 4000, i8* %4) #1
  %5 = bitcast i32* %b to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %5) #1
  %6 = bitcast i32* %d to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %6) #1
  %7 = bitcast [1000 x i32]* %aaa to i8*
  call void @llvm.lifetime.start.p0i8(i64 4000, i8* %7) #1
  %call = call i32 ([30 x i32]*, ...) bitcast (i32 (...)* @check_a to i32 ([30 x i32]*, ...)*)([30 x i32]* %a)
  %call1 = call i32 ([1000 x i32]*, ...) bitcast (i32 (...)* @bar1 to i32 ([1000 x i32]*, ...)*)([1000 x i32]* %aaa)
  call void asm sideeffect "", "~{esi},~{edi},~{ebp},~{ebx},~{rbx},~{r12},~{r13},~{r14},~{r15},~{rbp},~{dirflag},~{fpsr},~{flags}"() #1
  %call2 = call i32 ([30 x i32]*, ...) bitcast (i32 (...)* @bar1 to i32 ([30 x i32]*, ...)*)([30 x i32]* %a)
  %call3 = call i32 (i32*, ...) bitcast (i32 (...)* @check_f to i32 (i32*, ...)*)(i32* %f)
  call void asm sideeffect "", "~{esi},~{edi},~{ebp},~{ebx},~{rbx},~{r12},~{r13},~{r14},~{r15},~{rbp},~{dirflag},~{fpsr},~{flags}"() #1
  %call4 = call i32 ([30 x i32]*, ...) bitcast (i32 (...)* @bar1 to i32 ([30 x i32]*, ...)*)([30 x i32]* %a)
  %call5 = call i32 ([1000 x i32]*, [1000 x i32]*, [1000 x i32]*, ...) bitcast (i32 (...)* @bar3 to i32 ([1000 x i32]*, [1000 x i32]*, [1000 x i32]*, ...)*)([1000 x i32]* %aa, [1000 x i32]* %aaa, [1000 x i32]* %cc)
  call void asm sideeffect "", "~{esi},~{edi},~{ebp},~{ebx},~{rbx},~{r12},~{r13},~{r14},~{r15},~{rbp},~{dirflag},~{fpsr},~{flags}"() #1
  %call6 = call i32 ([30 x i32]*, [1000 x i32]*, ...) bitcast (i32 (...)* @bar2 to i32 ([30 x i32]*, [1000 x i32]*, ...)*)([30 x i32]* %a, [1000 x i32]* %cc)
  %call7 = call i32 (i32*, ...) bitcast (i32 (...)* @check_b to i32 (i32*, ...)*)(i32* %b)
  call void asm sideeffect "", "~{esi},~{edi},~{ebp},~{ebx},~{rbx},~{r12},~{r13},~{r14},~{r15},~{rbp},~{dirflag},~{fpsr},~{flags}"() #1
  %call8 = call i32 ([30 x i32]*, ...) bitcast (i32 (...)* @bar1 to i32 ([30 x i32]*, ...)*)([30 x i32]* %a)
  call void asm sideeffect "", "~{esi},~{edi},~{ebp},~{ebx},~{rbx},~{r12},~{r13},~{r14},~{r15},~{rbp},~{dirflag},~{fpsr},~{flags}"() #1
  %call9 = call i32 ([30 x i32]*, i32*, ...) bitcast (i32 (...)* @bar2 to i32 ([30 x i32]*, i32*, ...)*)([30 x i32]* %a, i32* %f)
  %call10 = call i32 (i32*, ...) bitcast (i32 (...)* @check_e to i32 (i32*, ...)*)(i32* %e)
  call void asm sideeffect "", "~{esi},~{edi},~{ebp},~{ebx},~{rbx},~{r12},~{r13},~{r14},~{r15},~{rbp},~{dirflag},~{fpsr},~{flags}"() #1
  %call11 = call i32 ([30 x i32]*, ...) bitcast (i32 (...)* @bar1 to i32 ([30 x i32]*, ...)*)([30 x i32]* %a)
  call void asm sideeffect "", "~{esi},~{edi},~{ebp},~{ebx},~{rbx},~{r12},~{r13},~{r14},~{r15},~{rbp},~{dirflag},~{fpsr},~{flags}"() #1
  %call12 = call i32 (i32*, i32*, ...) bitcast (i32 (...)* @bar2 to i32 (i32*, i32*, ...)*)(i32* %e, i32* %f)
  %call13 = call i32 (i32*, ...) bitcast (i32 (...)* @check_d to i32 (i32*, ...)*)(i32* %d)
  %call14 = call i32 ([30 x i32]*, ...) bitcast (i32 (...)* @bar1 to i32 ([30 x i32]*, ...)*)([30 x i32]* %a)
  call void asm sideeffect "", "~{esi},~{edi},~{ebp},~{ebx},~{rbx},~{r12},~{r13},~{r14},~{r15},~{rbp},~{dirflag},~{fpsr},~{flags}"() #1
  %call15 = call i32 (i32*, i32*, i32*, ...) bitcast (i32 (...)* @bar3 to i32 (i32*, i32*, i32*, ...)*)(i32* %d, i32* %e, i32* %f)
  call void asm sideeffect "", "~{esi},~{edi},~{ebp},~{ebx},~{rbx},~{r12},~{r13},~{r14},~{r15},~{rbp},~{dirflag},~{fpsr},~{flags}"() #1
  %call16 = call i32 ([30 x i32]*, ...) bitcast (i32 (...)* @bar1 to i32 ([30 x i32]*, ...)*)([30 x i32]* %a)
  call void @llvm.lifetime.end.p0i8(i64 4000, i8* %7) #1
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %6) #1
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %5) #1
  call void @llvm.lifetime.end.p0i8(i64 4000, i8* %4) #1
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %3) #1
  call void @llvm.lifetime.end.p0i8(i64 4000, i8* %2) #1
  call void @llvm.lifetime.end.p0i8(i64 120, i8* %1) #1
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %0) #1
  ret void
}

; Function Attrs: nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

declare i32 @check_a(...) #2
declare i32 @bar1(...) #2
declare i32 @check_f(...) #2
declare i32 @bar3(...) #2
declare i32 @bar2(...) #2
declare i32 @check_b(...) #2
declare i32 @check_e(...) #2
declare i32 @check_d(...) #2

; Function Attrs: nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

