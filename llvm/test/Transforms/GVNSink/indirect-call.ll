; RUN: opt < %s -gvn-sink -simplifycfg -hoist-common-insts=true -simplifycfg-sink-common=false -S | FileCheck %s

declare i8 @ext(i1)

define zeroext i1 @test1(i1 zeroext %flag, i32 %blksA, i32 %blksB, i32 %nblks, i8(i1)* %ext) {
entry:
  %cmp = icmp uge i32 %blksA, %nblks
  br i1 %flag, label %if.then, label %if.else

; CHECK-LABEL: test1
; CHECK: call i8 @ext
; CHECK: call i8 %ext
if.then:
  %frombool1 = call i8 @ext(i1 %cmp)
  br label %if.end

if.else:
  %frombool3 = call i8 %ext(i1 %cmp)
  br label %if.end

if.end:
  %obeys.0 = phi i8 [ %frombool1, %if.then ], [ %frombool3, %if.else ]
  %tobool4 = icmp ne i8 %obeys.0, 0
  ret i1 %tobool4
}

define zeroext i1 @test2(i1 zeroext %flag, i32 %blksA, i32 %blksB, i32 %nblks, i8(i1)* %ext) {
entry:
  %cmp = icmp uge i32 %blksA, %nblks
  br i1 %flag, label %if.then, label %if.else

; CHECK-LABEL: test2
; CHECK: call i8 %ext
; CHECK-NOT: call
if.then:
  %frombool1 = call i8 %ext(i1 %cmp)
  br label %if.end

if.else:
  %frombool3 = call i8 %ext(i1 %cmp)
  br label %if.end

if.end:
  %obeys.0 = phi i8 [ %frombool1, %if.then ], [ %frombool3, %if.else ]
  %tobool4 = icmp ne i8 %obeys.0, 0
  ret i1 %tobool4
}

define zeroext i1 @test3(i1 zeroext %flag, i32 %blksA, i32 %blksB, i32 %nblks, i8(i1)* %ext1, i8(i1)* %ext2) {
entry:
  %cmp = icmp uge i32 %blksA, %nblks
  br i1 %flag, label %if.then, label %if.else

; CHECK-LABEL: test3
; CHECK: %[[x:.*]] = select i1 %flag, i8 (i1)* %ext1, i8 (i1)* %ext2
; CHECK: call i8 %[[x]](i1 %cmp)
; CHECK-NOT: call
if.then:
  %frombool1 = call i8 %ext1(i1 %cmp)
  br label %if.end

if.else:
  %frombool3 = call i8 %ext2(i1 %cmp)
  br label %if.end

if.end:
  %obeys.0 = phi i8 [ %frombool1, %if.then ], [ %frombool3, %if.else ]
  %tobool4 = icmp ne i8 %obeys.0, 0
  ret i1 %tobool4
}

; Make sure no indirect call is introduced from direct calls
declare i8 @ext2(i1)
define zeroext i1 @test4(i1 zeroext %flag, i32 %blksA, i32 %blksB, i32 %nblks) {
entry:
  %cmp = icmp uge i32 %blksA, %nblks
  br i1 %flag, label %if.then, label %if.else

; CHECK-LABEL: test4
; CHECK: call i8 @ext(
; CHECK: call i8 @ext2(
if.then:
  %frombool1 = call i8 @ext(i1 %cmp)
  br label %if.end

if.else:
  %frombool3 = call i8 @ext2(i1 %cmp)
  br label %if.end

if.end:
  %obeys.0 = phi i8 [ %frombool1, %if.then ], [ %frombool3, %if.else ]
  %tobool4 = icmp ne i8 %obeys.0, 0
  ret i1 %tobool4
}
