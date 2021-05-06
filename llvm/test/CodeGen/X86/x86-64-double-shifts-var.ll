; RUN: llc < %s -mtriple=x86_64-- -mcpu=k8 | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-- -mcpu=opteron | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-- -mcpu=athlon64 | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-- -mcpu=athlon-fx | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-- -mcpu=k8-sse3 | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-- -mcpu=opteron-sse3 | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-- -mcpu=athlon64-sse3 | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-- -mcpu=amdfam10 | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-- -mcpu=btver1 | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-- -mcpu=btver2 | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-- -mcpu=bdver1 | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-- -mcpu=bdver2 | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-- -mcpu=bdver3 | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-- -mcpu=bdver4 | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-- -mcpu=znver1 | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-- -mcpu=znver2 | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-- -mcpu=znver3 | FileCheck %s

; Verify that for the X86_64 processors that are known to have poor latency
; double precision shift instructions we do not generate 'shld' or 'shrd'
; instructions.

;uint64_t lshift(uint64_t a, uint64_t b, int c)
;{
;    return (a << c) | (b >> (64-c));
;}

define i64 @lshift(i64 %a, i64 %b, i32 %c) nounwind readnone {
entry:
; CHECK-NOT: shld
  %sh_prom = zext i32 %c to i64
  %shl = shl i64 %a, %sh_prom
  %sub = sub nsw i32 64, %c
  %sh_prom1 = zext i32 %sub to i64
  %shr = lshr i64 %b, %sh_prom1
  %or = or i64 %shr, %shl
  ret i64 %or
}

;uint64_t rshift(uint64_t a, uint64_t b, int c)
;{
;    return (a >> c) | (b << (64-c));
;}

define i64 @rshift(i64 %a, i64 %b, i32 %c) nounwind readnone {
entry:
; CHECK-NOT: shrd
  %sh_prom = zext i32 %c to i64
  %shr = lshr i64 %a, %sh_prom
  %sub = sub nsw i32 64, %c
  %sh_prom1 = zext i32 %sub to i64
  %shl = shl i64 %b, %sh_prom1
  %or = or i64 %shl, %shr
  ret i64 %or
}


