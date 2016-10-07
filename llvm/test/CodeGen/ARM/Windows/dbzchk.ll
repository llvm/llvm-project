; RUN: llc -mtriple thumbv7--windows-itanium -print-machineinstrs=expand-isel-pseudos -verify-machineinstrs -o /dev/null %s 2>&1 | FileCheck %s -check-prefix CHECK-DIV

; int f(int n, int d) {
;   if (n / d)
;     return 1;
;   return 0;
; }

define arm_aapcs_vfpcc i32 @f(i32 %n, i32 %d) {
entry:
  %retval = alloca i32, align 4
  %n.addr = alloca i32, align 4
  %d.addr = alloca i32, align 4
  store i32 %n, i32* %n.addr, align 4
  store i32 %d, i32* %d.addr, align 4
  %0 = load i32, i32* %n.addr, align 4
  %1 = load i32, i32* %d.addr, align 4
  %div = sdiv i32 %0, %1
  %tobool = icmp ne i32 %div, 0
  br i1 %tobool, label %if.then, label %if.end

if.then:
  store i32 1, i32* %retval, align 4
  br label %return

if.end:
  store i32 0, i32* %retval, align 4
  br label %return

return:
  %2 = load i32, i32* %retval, align 4
  ret i32 %2
}

; CHECK-DIV-DAG: BB#0
; CHECK-DIV-DAG: Successors according to CFG: BB#5({{.*}}) BB#4
; CHECK-DIV-DAG: BB#1
; CHECK-DIV-DAG: Successors according to CFG: BB#3
; CHECK-DIV-DAG: BB#2
; CHECK-DIV-DAG: Successors according to CFG: BB#3
; CHECK-DIV-DAG: BB#3
; CHECK-DIV-DAG: BB#4
; CHECK-DIV-DAG: Successors according to CFG: BB#1({{.*}}) BB#2
; CHECK-DIV-DAG: BB#5

; RUN: llc -mtriple thumbv7--windows-itanium -print-machineinstrs=expand-isel-pseudos -verify-machineinstrs -o /dev/null %s 2>&1 | FileCheck %s -check-prefix CHECK-MOD

; int r;
; int g(int l, int m) {
;   if (m <= 0)
;     return 0;
;   return (r = l % m);
; }

@r = common global i32 0, align 4

define arm_aapcs_vfpcc i32 @g(i32 %l, i32 %m) {
entry:
  %cmp = icmp eq i32 %m, 0
  br i1 %cmp, label %return, label %if.end

if.end:
  %rem = urem i32 %l, %m
  store i32 %rem, i32* @r, align 4
  br label %return

return:
  %retval.0 = phi i32 [ %rem, %if.end ], [ 0, %entry ]
  ret i32 %retval.0
}

; CHECK-MOD-DAG: BB#0
; CHECK-MOD-DAG: Successors according to CFG: BB#2({{.*}}) BB#1
; CHECK-MOD-DAG: BB#1
; CHECK-MOD-DAG: Successors according to CFG: BB#4({{.*}}) BB#3
; CHECK-MOD-DAG: BB#2
; CHECK-MOD-DAG: BB#3
; CHECK-MOD-DAG: Successors according to CFG: BB#2
; CHECK-MOD-DAG: BB#4

; RUN: llc -mtriple thumbv7--windows-itanium -print-machineinstrs=expand-isel-pseudos -verify-machineinstrs -filetype asm -o /dev/null %s 2>&1 | FileCheck %s -check-prefix CHECK-CFG
; RUN: llc -mtriple thumbv7--windows-itanium -print-machineinstrs=expand-isel-pseudos -verify-machineinstrs -filetype asm -o - %s | FileCheck %s -check-prefix CHECK-CFG-ASM

; unsigned c;
; extern unsigned long g(void);
; int f(unsigned u, signed char b) {
;   if (b)
;     c = g() % u;
;   return c;
; }

@c = common global i32 0, align 4

declare arm_aapcs_vfpcc i32 @i()

define arm_aapcs_vfpcc i32 @h(i32 %u, i8 signext %b) #0 {
entry:
  %tobool = icmp eq i8 %b, 0
  br i1 %tobool, label %entry.if.end_crit_edge, label %if.then

entry.if.end_crit_edge:
  %.pre = load i32, i32* @c, align 4
  br label %if.end

if.then:
  %call = tail call arm_aapcs_vfpcc i32 @i()
  %rem = urem i32 %call, %u
  store i32 %rem, i32* @c, align 4
  br label %if.end

if.end:
  %0 = phi i32 [ %.pre, %entry.if.end_crit_edge ], [ %rem, %if.then ]
  ret i32 %0
}

attributes #0 = { optsize }

; CHECK-CFG-DAG: BB#0
; CHECK-CFG-DAG: t2Bcc <BB#2>
; CHECK-CFG-DAG: t2B <BB#1>

; CHECK-CFG-DAG: BB#1
; CHECK-CFG-DAG: t2B <BB#3>

; CHECK-CFG-DAG: BB#2
; CHECK-CFG-DAG: tCBZ %vreg{{[0-9]}}, <BB#5>
; CHECK-CFG-DAG: t2B <BB#4>

; CHECK-CFG-DAG: BB#4

; CHECK-CFG-DAG: BB#3
; CHECK-CFG-DAG: tBX_RET

; CHECK-CFG-DAG: BB#5
; CHECK-CFG-DAG: t2UDF 249

; CHECK-CFG-ASM-LABEL: h:
; CHECK-CFG-ASM: cbz r{{[0-9]}}, .LBB2_2
; CHECK-CFG-ASM: b .LBB2_4
; CHECK-CFG-ASM-LABEL: .LBB2_2:
; CHECK-CFG-ASM-NEXT: udf.w #249
; CHECK-CFG-ASM-LABEL: .LBB2_4:
; CHECK-CFG-ASM: bl __rt_udiv
; CHECK-CFG-ASM: pop.w {{{.*}}, r11, pc}

; RUN: llc -O0 -mtriple thumbv7--windows-itanium -verify-machineinstrs -filetype asm -o - %s | FileCheck %s -check-prefix CHECK-WIN__DBZCHK

; long k(void);
; int l(void);
; int j(int i) {
;   if (l() == -1)
;     return 0;
;   return k() % i;
; }

declare arm_aapcs_vfpcc i32 @k()
declare arm_aapcs_vfpcc i32 @l()

define arm_aapcs_vfpcc i32 @j(i32 %i) {
entry:
  %retval = alloca i32, align 4
  %i.addr = alloca i32, align 4
  store i32 %i, i32* %i.addr, align 4
  %call = call arm_aapcs_vfpcc i32 @l()
  %cmp = icmp eq i32 %call, -1
  br i1 %cmp, label %if.then, label %if.end

if.then:
  store i32 0, i32* %retval, align 4
  br label %return

if.end:
  %call1 = call arm_aapcs_vfpcc i32 @k()
  %0 = load i32, i32* %i.addr, align 4
  %rem = srem i32 %call1, %0
  store i32 %rem, i32* %retval, align 4
  br label %return

return:
  %1 = load i32, i32* %retval, align 4
  ret i32 %1
}

; CHECK-WIN__DBZCHK-LABEL: j:
; CHECK-WIN__DBZCHK: cbz r{{[0-7]}}, .LBB
; CHECK-WIN__DBZCHK-NOT: cbz r8, .LBB
; CHECK-WIN__DBZCHK-NOT: cbz r9, .LBB
; CHECK-WIN__DBZCHK-NOT: cbz r10, .LBB
; CHECK-WIN__DBZCHK-NOT: cbz r11, .LBB
; CHECK-WIN__DBZCHK-NOT: cbz ip, .LBB
; CHECK-WIN__DBZCHK-NOT: cbz lr, .LBB

