; RUN: llc -verify-machineinstrs -mcpu=pwr8 < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

@zeroEqualityTest01.buffer1 = private unnamed_addr constant [3 x i32] [i32 1, i32 2, i32 4], align 4
@zeroEqualityTest01.buffer2 = private unnamed_addr constant [3 x i32] [i32 1, i32 2, i32 3], align 4
@zeroEqualityTest02.buffer1 = private unnamed_addr constant [4 x i32] [i32 4, i32 0, i32 0, i32 0], align 4
@zeroEqualityTest02.buffer2 = private unnamed_addr constant [4 x i32] [i32 3, i32 0, i32 0, i32 0], align 4
@zeroEqualityTest03.buffer1 = private unnamed_addr constant [4 x i32] [i32 0, i32 0, i32 0, i32 3], align 4
@zeroEqualityTest03.buffer2 = private unnamed_addr constant [4 x i32] [i32 0, i32 0, i32 0, i32 4], align 4
@zeroEqualityTest04.buffer1 = private unnamed_addr constant [15 x i32] [i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14], align 4
@zeroEqualityTest04.buffer2 = private unnamed_addr constant [15 x i32] [i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 13], align 4

; Function Attrs: nounwind readonly
declare signext i32 @memcmp(i8* nocapture, i8* nocapture, i64) local_unnamed_addr #1

; Validate with if(memcmp())
; Function Attrs: nounwind readonly
define signext i32 @zeroEqualityTest01() local_unnamed_addr #0 {
entry:
  %call = tail call signext i32 @memcmp(i8* bitcast ([3 x i32]* @zeroEqualityTest01.buffer1 to i8*), i8* bitcast ([3 x i32]* @zeroEqualityTest01.buffer2 to i8*), i64 16)
  %not.tobool = icmp ne i32 %call, 0
  %. = zext i1 %not.tobool to i32
  ret i32 %.

  ; CHECK-LABEL: @zeroEqualityTest01
  ; CHECK-LABEL: %res_block
  ; CHECK: li 3, 1
  ; CHECK-NEXT: clrldi
  ; CHECK-NEXT: blr
  ; CHECK: li 3, 0
  ; CHECK-NEXT: clrldi
  ; CHECK-NEXT: blr
}

; Validate with if(memcmp() == 0)
; Function Attrs: nounwind readonly
define signext i32 @zeroEqualityTest02() local_unnamed_addr #0 {
entry:
  %call = tail call signext i32 @memcmp(i8* bitcast ([4 x i32]* @zeroEqualityTest02.buffer1 to i8*), i8* bitcast ([4 x i32]* @zeroEqualityTest02.buffer2 to i8*), i64 16)
  %not.cmp = icmp ne i32 %call, 0
  %. = zext i1 %not.cmp to i32
  ret i32 %.

  ; CHECK-LABEL: @zeroEqualityTest02
  ; CHECK-LABEL: %res_block
  ; CHECK: li 3, 1
  ; CHECK-NEXT: clrldi
  ; CHECK-NEXT: blr
  ; CHECK: li 3, 0
  ; CHECK-NEXT: clrldi
  ; CHECK-NEXT: blr
}

; Validate with > 0
; Function Attrs: nounwind readonly
define signext i32 @zeroEqualityTest03() local_unnamed_addr #0 {
entry:
  %call = tail call signext i32 @memcmp(i8* bitcast ([4 x i32]* @zeroEqualityTest02.buffer1 to i8*), i8* bitcast ([4 x i32]* @zeroEqualityTest02.buffer2 to i8*), i64 16)
  %not.cmp = icmp slt i32 %call, 1
  %. = zext i1 %not.cmp to i32
  ret i32 %.

  ; CHECK-LABEL: @zeroEqualityTest03
  ; CHECK-LABEL: %res_block
  ; CHECK: cmpld
  ; CHECK-NEXT: li [[LI:[0-9]+]], 1
  ; CHECK-NEXT: li [[LI2:[0-9]+]], -1
  ; CHECK-NEXT: isel [[ISEL:[0-9]+]], [[LI2]], [[LI]], 0
}

; Validate with < 0
; Function Attrs: nounwind readonly
define signext i32 @zeroEqualityTest04() local_unnamed_addr #0 {
entry:
  %call = tail call signext i32 @memcmp(i8* bitcast ([4 x i32]* @zeroEqualityTest03.buffer1 to i8*), i8* bitcast ([4 x i32]* @zeroEqualityTest03.buffer2 to i8*), i64 16)
  %call.lobit = lshr i32 %call, 31
  %call.lobit.not = xor i32 %call.lobit, 1
  ret i32 %call.lobit.not

  ; CHECK-LABEL: @zeroEqualityTest04
  ; CHECK-LABEL: %res_block
  ; CHECK: cmpld
  ; CHECK-NEXT: li [[LI:[0-9]+]], 1
  ; CHECK-NEXT: li [[LI2:[0-9]+]], -1
  ; CHECK-NEXT: isel [[ISEL:[0-9]+]], [[LI2]], [[LI]], 0
}

; Validate with memcmp()?:
; Function Attrs: nounwind readonly
define signext i32 @zeroEqualityTest05() local_unnamed_addr #0 {
entry:
  %call = tail call signext i32 @memcmp(i8* bitcast ([15 x i32]* @zeroEqualityTest04.buffer1 to i8*), i8* bitcast ([15 x i32]* @zeroEqualityTest04.buffer2 to i8*), i64 16)
  %not.tobool = icmp eq i32 %call, 0
  %cond = zext i1 %not.tobool to i32
  ret i32 %cond

  ; CHECK-LABEL: @zeroEqualityTest05
  ; CHECK-LABEL: %res_block
  ; CHECK: li 3, 1
  ; CHECK: li 3, 0
}

; Validate with !memcmp()?:
; Function Attrs: nounwind readonly
define signext i32 @zeroEqualityTest06() local_unnamed_addr #0 {
entry:
  %call = tail call signext i32 @memcmp(i8* bitcast ([15 x i32]* @zeroEqualityTest04.buffer1 to i8*), i8* bitcast ([15 x i32]* @zeroEqualityTest04.buffer2 to i8*), i64 16)
  %not.lnot = icmp ne i32 %call, 0
  %cond = zext i1 %not.lnot to i32
  ret i32 %cond

  ; CHECK-LABEL: @zeroEqualityTest06
  ; CHECK-LABEL: %res_block
  ; CHECK: li 3, 1
  ; CHECK-NEXT: clrldi
  ; CHECK-NEXT: blr
  ; CHECK: li 3, 0
  ; CHECK-NEXT: clrldi
  ; CHECK-NEXT: blr
}
