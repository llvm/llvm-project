; RUN: llc < %s -mattr=+reserve-r6 -mtriple=arm-linux-gnueabi -O0 -filetype=asm --regalloc=fast 2>&1 | FileCheck %s
;
; Equivalent C source code
; register unsigned r6 asm("r6");
; void bar(unsigned int i,
;          unsigned int j,
;          unsigned int k,
;          unsigned int l,
;          unsigned int m,
;          unsigned int n,
;          unsigned int o,
;          unsigned int p)
; {
;     r6 = 10;
;     unsigned int result = i + j + k + l + m + n + o + p;
; }
declare void @llvm.write_register.i32(metadata, i32) nounwind

define void @bar(i32 %i, i32 %j, i32 %k, i32 %l, i32 %m, i32 %n, i32 %o, i32 %p) nounwind {
entry:
; CHECK-NOT: push {{{.*}}r6,{{.*}}}
; CHECK: {{.*}}mov{{.*}}r6,{{.*}}
; CHECK-NOT: {{.*}}r6{{.*}}
  %i.addr = alloca i32, align 4
  %j.addr = alloca i32, align 4
  %k.addr = alloca i32, align 4
  %l.addr = alloca i32, align 4
  %m.addr = alloca i32, align 4
  %n.addr = alloca i32, align 4
  %o.addr = alloca i32, align 4
  %p.addr = alloca i32, align 4
  %result = alloca i32, align 4
  store i32 %i, i32* %i.addr, align 4
  store i32 %j, i32* %j.addr, align 4
  store i32 %k, i32* %k.addr, align 4
  store i32 %l, i32* %l.addr, align 4
  store i32 %m, i32* %m.addr, align 4
  store i32 %n, i32* %n.addr, align 4
  store i32 %o, i32* %o.addr, align 4
  store i32 %p, i32* %p.addr, align 4
  call void @llvm.write_register.i32(metadata !0, i32 10)
  %0 = load i32, i32* %i.addr, align 4
  %1 = load i32, i32* %j.addr, align 4
  %add = add i32 %0, %1
  %2 = load i32, i32* %k.addr, align 4
  %add1 = add i32 %add, %2
  %3 = load i32, i32* %l.addr, align 4
  %add2 = add i32 %add1, %3
  %4 = load i32, i32* %m.addr, align 4
  %add3 = add i32 %add2, %4
  %5 = load i32, i32* %n.addr, align 4
  %add4 = add i32 %add3, %5
  %6 = load i32, i32* %o.addr, align 4
  %add5 = add i32 %add4, %6
  %7 = load i32, i32* %p.addr, align 4
  %add6 = add i32 %add5, %7
  store i32 %add6, i32* %result, align 4
  ret void
}

!llvm.named.register.r6 = !{!0}
!0 = !{!"r6"}

