; RUN: llc < %s -mtriple=arm-linux -mcpu=generic | FileCheck %s

define i32 @uadd_overflow(i32 %a, i32 %b) #0 {
  %sadd = tail call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 %a, i32 %b)
  %1 = extractvalue { i32, i1 } %sadd, 1
  %2 = zext i1 %1 to i32
  ret i32 %2

  ; CHECK-LABEL: uadd_overflow:
  ; CHECK: add r[[R2:[0-9]+]], r[[R0:[0-9]+]], r[[R1:[0-9]+]]
  ; CHECK: mov r[[R1]], #1
  ; CHECK: cmp r[[R2]], r[[R0]]
  ; CHECK: movhs r[[R1]], #0
}


define i32 @sadd_overflow(i32 %a, i32 %b) #0 {
  %sadd = tail call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %a, i32 %b)
  %1 = extractvalue { i32, i1 } %sadd, 1
  %2 = zext i1 %1 to i32
  ret i32 %2

  ; CHECK-LABEL: sadd_overflow:
  ; CHECK: add r[[R2:[0-9]+]], r[[R0:[0-9]+]], r[[R1:[0-9]+]]
  ; CHECK: mov r[[R1]], #1
  ; CHECK: cmp r[[R2]], r[[R0]]
  ; CHECK: movvc r[[R1]], #0
}

define i32 @usub_overflow(i32 %a, i32 %b) #0 {
  %sadd = tail call { i32, i1 } @llvm.usub.with.overflow.i32(i32 %a, i32 %b)
  %1 = extractvalue { i32, i1 } %sadd, 1
  %2 = zext i1 %1 to i32
  ret i32 %2

  ; CHECK-LABEL: usub_overflow:
  ; CHECK: mov r[[R2]], #1
  ; CHECK: cmp r[[R0]], r[[R1]]
  ; CHECK: movhs r[[R2]], #0
}

define i32 @ssub_overflow(i32 %a, i32 %b) #0 {
  %sadd = tail call { i32, i1 } @llvm.ssub.with.overflow.i32(i32 %a, i32 %b)
  %1 = extractvalue { i32, i1 } %sadd, 1
  %2 = zext i1 %1 to i32
  ret i32 %2

  ; CHECK-LABEL: ssub_overflow:
  ; CHECK: mov r[[R2]], #1
  ; CHECK: cmp r[[R0]], r[[R1]]
  ; CHECK: movvc r[[R2]], #0
}

declare { i32, i1 } @llvm.uadd.with.overflow.i32(i32, i32) #1
declare { i32, i1 } @llvm.sadd.with.overflow.i32(i32, i32) #2
declare { i32, i1 } @llvm.usub.with.overflow.i32(i32, i32) #3
declare { i32, i1 } @llvm.ssub.with.overflow.i32(i32, i32) #4
