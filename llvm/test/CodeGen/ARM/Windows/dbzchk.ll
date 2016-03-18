; RUN: llc -mtriple thumbv7--windows-itanium -print-machineinstrs=expand-isel-pseudos -o /dev/null %s 2>&1 | FileCheck %s -check-prefix CHECK-DIV

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

; RUN: llc -mtriple thumbv7--windows-itanium -print-machineinstrs=expand-isel-pseudos -o /dev/null %s 2>&1 | FileCheck %s -check-prefix CHECK-MOD

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

