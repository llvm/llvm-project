; RUN: llc -mtriple=thumbv7-- -mcpu=cortex-a9 -O1 -filetype=obj %s -o - | llvm-objdump --triple=thumbv7-- --mcpu=cortex-a9 -d - | FileCheck %s

target datalayout = "e-m:e-p:32:32-i1:8:32-i8:8:32-i16:16:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv7--linux-gnueabi"

declare ptr @llvm.returnaddress(i32)

define ptr @wrong-t2stmia-size-reduction(ptr %addr, i32 %val0, i32 %val1) minsize {
  store i32 %val0, ptr %addr
  %addr1 = getelementptr i32, ptr %addr, i32 1
  %addr2 = getelementptr i32, ptr %addr, i32 2
  %lr = call ptr @llvm.returnaddress(i32 0)
  %lr32 = ptrtoint ptr %lr to i32
  store i32 %val1, ptr %addr1
  store i32 %lr32, ptr %addr2

  %addr3 = getelementptr i32, ptr %addr, i32 3
  ret ptr %addr3
}

; Check that stm writes three registers.  The bug caused one of registers (LR,
; which invalid for Thumb1 form of STMIA instruction) to be dropped.
; CHECK-LABEL: <wrong-t2stmia-size-reduction>:
; CHECK: stm{{[^,]*}}, {{{.*,.*,.*}}}
