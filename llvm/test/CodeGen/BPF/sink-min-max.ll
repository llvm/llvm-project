; RUN: opt --bpf-check-and-opt-ir -S -mtriple=bpf-pc-linux %s | FileCheck %s

; Test plan:
; @test1: x <  umin(i64 a, i64 b)
; @test2: x <  umax(i64 a, i64 b)
; @test3: x >= umin(i64 a, i64 b)
; @test4: x >= umax(i64 a, i64 b)
; @test5: umin(i64 a, i64 b) >= x
; @test6: x <  smin(i64 a, i64 b)
; @test7: x <  umin(i32 a, i32 b)
; @test8: x <  zext i64 umin(i32 a, i32 b)
; @test9: x <  sext i64 umin(i32 a, i32 b)
; @test10: check that umin belonging to the same loop is not touched
; @test11: check that nested loops are processed

define i32 @test1(i64 %a, i64 %b, i64 %x) {
entry:
  %min = tail call i64 @llvm.umin.i64(i64 %a, i64 %b)
  br label %loop
loop:
  %cmp = icmp ult i64 %x, %min
  br i1 %cmp, label %loop, label %ret
ret: ret i32 0
}

; CHECK:       @test1
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %loop
; CHECK-EMPTY:
; CHECK-NEXT:  loop:
; CHECK-NEXT:    %0 = icmp ult i64 %x, %a
; CHECK-NEXT:    %1 = icmp ult i64 %x, %b
; CHECK-NEXT:    %2 = select i1 %0, i1 %1, i1 false
; CHECK-NEXT:    br i1 %2, label %loop, label %ret

define i32 @test2(i64 %a, i64 %b, i64 %x) {
entry:
  %max = tail call i64 @llvm.umax.i64(i64 %a, i64 %b)
  br label %loop
loop:
  %cmp = icmp ult i64 %x, %max
  br i1 %cmp, label %loop, label %ret
ret: ret i32 0
}

; CHECK:       @test2
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %loop
; CHECK-EMPTY:
; CHECK-NEXT:  loop:
; CHECK-NEXT:    %0 = icmp ult i64 %x, %a
; CHECK-NEXT:    %1 = icmp ult i64 %x, %b
; CHECK-NEXT:    %2 = select i1 %0, i1 true, i1 %1
; CHECK-NEXT:    br i1 %2, label %loop, label %ret

define i32 @test3(i64 %a, i64 %b, i64 %x) {
entry:
  %min = tail call i64 @llvm.umin.i64(i64 %a, i64 %b)
  br label %loop
loop:
  %cmp = icmp uge i64 %x, %min
  br i1 %cmp, label %loop, label %ret
ret: ret i32 0
}

; CHECK:       @test3
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %loop
; CHECK-EMPTY:
; CHECK-NEXT:  loop:
; CHECK-NEXT:    %0 = icmp uge i64 %x, %a
; CHECK-NEXT:    %1 = icmp uge i64 %x, %b
; CHECK-NEXT:    %2 = select i1 %0, i1 true, i1 %1
; CHECK-NEXT:    br i1 %2, label %loop, label %ret

define i32 @test4(i64 %a, i64 %b, i64 %x) {
entry:
  %max = tail call i64 @llvm.umax.i64(i64 %a, i64 %b)
  br label %loop
loop:
  %cmp = icmp uge i64 %x, %max
  br i1 %cmp, label %loop, label %ret
ret: ret i32 0
}

; CHECK:       @test4
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %loop
; CHECK-EMPTY:
; CHECK-NEXT:  loop:
; CHECK-NEXT:    %0 = icmp uge i64 %x, %a
; CHECK-NEXT:    %1 = icmp uge i64 %x, %b
; CHECK-NEXT:    %2 = select i1 %0, i1 %1, i1 false
; CHECK-NEXT:    br i1 %2, label %loop, label %ret

define i32 @test5(i64 %a, i64 %b, i64 %x) {
entry:
  %min = tail call i64 @llvm.umin.i64(i64 %a, i64 %b)
  br label %loop
loop:
  %cmp = icmp uge i64 %min, %x
  br i1 %cmp, label %loop, label %ret
ret: ret i32 0
}

; CHECK:       @test5
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %loop
; CHECK-EMPTY:
; CHECK-NEXT:  loop:
; CHECK:         %0 = icmp ule i64 %x, %a
; CHECK-NEXT:    %1 = icmp ule i64 %x, %b
; CHECK-NEXT:    %2 = select i1 %0, i1 %1, i1 false
; CHECK-NEXT:    br i1 %2, label %loop, label %ret

define i32 @test6(i64 %a, i64 %b, i64 %x) {
entry:
  %min = tail call i64 @llvm.smin.i64(i64 %a, i64 %b)
  br label %loop
loop:
  %cmp = icmp slt i64 %x, %min
  br i1 %cmp, label %loop, label %ret
ret: ret i32 0
}

; CHECK:       @test6
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %loop
; CHECK-EMPTY:
; CHECK-NEXT:  loop:
; CHECK:         %0 = icmp slt i64 %x, %a
; CHECK-NEXT:    %1 = icmp slt i64 %x, %b
; CHECK-NEXT:    %2 = select i1 %0, i1 %1, i1 false
; CHECK-NEXT:    br i1 %2, label %loop, label %ret

define i32 @test7(i32 %a, i32 %b, i32 %x) {
entry:
  %min = tail call i32 @llvm.umin.i32(i32 %a, i32 %b)
  br label %loop
loop:
  %cmp = icmp ult i32 %x, %min
  br i1 %cmp, label %loop, label %ret
ret: ret i32 0
}

; CHECK:       @test7
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %loop
; CHECK-EMPTY:
; CHECK-NEXT:  loop:
; CHECK:         %0 = icmp ult i32 %x, %a
; CHECK-NEXT:    %1 = icmp ult i32 %x, %b
; CHECK-NEXT:    %2 = select i1 %0, i1 %1, i1 false
; CHECK-NEXT:    br i1 %2, label %loop, label %ret

define i32 @test8(i32 %a, i32 %b, i64 %x) {
entry:
  %min = tail call i32 @llvm.umin.i32(i32 %a, i32 %b)
  br label %loop
loop:
  %ext = zext i32 %min to i64
  %cmp = icmp ult i64 %x, %ext
  br i1 %cmp, label %loop, label %ret
ret: ret i32 0
}

; CHECK:       @test8
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %loop
; CHECK-EMPTY:
; CHECK-NEXT:  loop:
; CHECK-NEXT:    %0 = zext i32 %a to i64
; CHECK-NEXT:    %1 = zext i32 %b to i64
; CHECK-NEXT:    %2 = icmp ult i64 %x, %0
; CHECK-NEXT:    %3 = icmp ult i64 %x, %1
; CHECK-NEXT:    %4 = select i1 %2, i1 %3, i1 false
; CHECK-NEXT:    br i1 %4, label %loop, label %ret

define i32 @test9(i32 %a, i32 %b, i64 %x) {
entry:
  %min = tail call i32 @llvm.umin.i32(i32 %a, i32 %b)
  br label %loop
loop:
  %ext = sext i32 %min to i64
  %cmp = icmp ult i64 %x, %ext
  br i1 %cmp, label %loop, label %ret
ret: ret i32 0
}

; CHECK:       @test9
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %loop
; CHECK-EMPTY:
; CHECK-NEXT:  loop:
; CHECK-NEXT:    %0 = sext i32 %a to i64
; CHECK-NEXT:    %1 = sext i32 %b to i64
; CHECK-NEXT:    %2 = icmp ult i64 %x, %0
; CHECK-NEXT:    %3 = icmp ult i64 %x, %1
; CHECK-NEXT:    %4 = select i1 %2, i1 %3, i1 false
; CHECK-NEXT:    br i1 %4, label %loop, label %ret

; umin within the loop body is unchanged
define i32 @test10(i64 %a, i64 %b, i64 %x) {
entry:
  br label %loop
loop:
  %min = tail call i64 @llvm.umin.i64(i64 %a, i64 %b)
  %cmp = icmp ult i64 %x, %min
  br i1 %cmp, label %loop, label %ret
ret: ret i32 0
}

; CHECK:       @test10
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %loop
; CHECK-EMPTY:
; CHECK-NEXT:  loop:
; CHECK-NEXT:    %min = tail call i64 @llvm.umin.i64(i64 %a, i64 %b)
; CHECK-NEXT:    %cmp = icmp ult i64 %x, %min
; CHECK-NEXT:    br i1 %cmp, label %loop, label %ret

; umin from outer loop body is processed
define i32 @test11(i64 %a, i64 %b, i64 %x) {
entry:
  br label %loop

loop:
  %min = tail call i64 @llvm.umin.i64(i64 %a, i64 %b)
  br label %nested.loop
nested.loop:
  %cmp = icmp ult i64 %x, %min
  br i1 %cmp, label %nested.loop, label %loop

ret: ret i32 0
}

; CHECK:       @test11
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %loop
; CHECK-EMPTY:
; CHECK-NEXT:  loop:
; CHECK-NEXT:    br label %nested.loop
; CHECK-EMPTY:
; CHECK-NEXT:  nested.loop:
; CHECK-NEXT:    %0 = icmp ult i64 %x, %a
; CHECK-NEXT:    %1 = icmp ult i64 %x, %b
; CHECK-NEXT:    %2 = select i1 %0, i1 %1, i1 false
; CHECK-NEXT:    br i1 %2, label %nested.loop, label %loop

declare i64 @llvm.umin.i64(i64, i64)
declare i64 @llvm.smin.i64(i64, i64)
declare i64 @llvm.umax.i64(i64, i64)
declare i64 @llvm.smax.i64(i64, i64)

declare i32 @llvm.umin.i32(i32, i32)
declare i32 @llvm.smin.i32(i32, i32)
declare i32 @llvm.umax.i32(i32, i32)
declare i32 @llvm.smax.i32(i32, i32)
