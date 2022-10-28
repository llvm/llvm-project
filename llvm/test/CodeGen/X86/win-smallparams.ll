; When we accept small parameters on Windows, make sure we do not assume they
; are zero or sign extended in memory or in registers.

; RUN: llc < %s -mtriple=x86_64-windows-msvc | FileCheck %s --check-prefix=WIN64
; RUN: llc < %s -mtriple=x86_64-windows-gnu | FileCheck %s --check-prefix=WIN64
; RUN: llc < %s -mtriple=i686-windows-msvc | FileCheck %s --check-prefix=WIN32
; RUN: llc < %s -mtriple=i686-windows-gnu | FileCheck %s --check-prefix=WIN32

define void @call() {
entry:
  %rv = call i32 @manyargs(i8 1, i16 2, i8 3, i16 4, i8 5, i16 6)
  ret void
}

define i32 @manyargs(i8 %a, i16 %b, i8 %c, i16 %d, i8 %e, i16 %f) {
entry:
  %aa = sext i8 %a to i32
  %bb = sext i16 %b to i32
  %cc = zext i8 %c to i32
  %dd = zext i16 %d to i32
  %ee = zext i8 %e to i32
  %ff = zext i16 %f to i32
  %t0 = add i32 %aa, %bb
  %t1 = add i32 %t0, %cc
  %t2 = add i32 %t1, %dd
  %t3 = add i32 %t2, %ee
  %t4 = add i32 %t3, %ff
  ret i32 %t4
}

; WIN64-LABEL: call:
; WIN64-DAG: movw $6, 40(%rsp)
; WIN64-DAG: movb $5, 32(%rsp)
; WIN64-DAG: movb $1, %cl
; WIN64-DAG: movw $2, %dx
; WIN64-DAG: movb $3, %r8b
; WIN64-DAG: movw $4, %r9w
; WIN64: callq manyargs

; WIN64-LABEL: manyargs:
; WIN64-DAG: movsbl %cl,
; WIN64-DAG: movswl %dx,
; WIN64-DAG: movzbl %r8b,
; WIN64-DAG: movzwl %r9w,
; WIN64-DAG: movzbl 40(%rsp),
; WIN64-DAG: movzwl 48(%rsp),
; WIN64: retq


; WIN32-LABEL: _call:
; WIN32: pushl $6
; WIN32: pushl $5
; WIN32: pushl $4
; WIN32: pushl $3
; WIN32: pushl $2
; WIN32: pushl $1
; WIN32: calll _manyargs

; WIN32-LABEL: _manyargs:
; WIN32: pushl %edi
; WIN32: pushl %esi
; WIN32-DAG: movzwl 32(%esp),
; WIN32-DAG: movzbl 28(%esp),
; WIN32-DAG: movzwl 24(%esp),
; WIN32-DAG: movzbl 20(%esp),
; WIN32-DAG: movswl 16(%esp),
; WIN32-DAG: movsbl 12(%esp),
; WIN32: retl

