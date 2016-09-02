; RUN: opt < %s -simplifycfg -S | FileCheck %s

define zeroext i1 @test1(i1 zeroext %flag, i32 %blksA, i32 %blksB, i32 %nblks) {
entry:
  br i1 %flag, label %if.then, label %if.else

; CHECK-LABEL: test1
; CHECK: add
; CHECK: select
; CHECK: icmp
; CHECK-NOT: br
if.then:
  %cmp = icmp uge i32 %blksA, %nblks
  %frombool1 = zext i1 %cmp to i8
  br label %if.end

if.else:
  %add = add i32 %nblks, %blksB
  %cmp2 = icmp ule i32 %add, %blksA
  %frombool3 = zext i1 %cmp2 to i8
  br label %if.end

if.end:
  %obeys.0 = phi i8 [ %frombool1, %if.then ], [ %frombool3, %if.else ]
  %tobool4 = icmp ne i8 %obeys.0, 0
  ret i1 %tobool4
}

define zeroext i1 @test2(i1 zeroext %flag, i32 %blksA, i32 %blksB, i32 %nblks) {
entry:
  br i1 %flag, label %if.then, label %if.else

; CHECK-LABEL: test2
; CHECK: add
; CHECK: select
; CHECK: icmp
; CHECK-NOT: br
if.then:
  %cmp = icmp uge i32 %blksA, %nblks
  %frombool1 = zext i1 %cmp to i8
  br label %if.end

if.else:
  %add = add i32 %nblks, %blksB
  %cmp2 = icmp uge i32 %blksA, %add
  %frombool3 = zext i1 %cmp2 to i8
  br label %if.end

if.end:
  %obeys.0 = phi i8 [ %frombool1, %if.then ], [ %frombool3, %if.else ]
  %tobool4 = icmp ne i8 %obeys.0, 0
  ret i1 %tobool4
}

declare i32 @foo(i32, i32) nounwind readnone

define i32 @test3(i1 zeroext %flag, i32 %x, i32 %y) {
entry:
  br i1 %flag, label %if.then, label %if.else

if.then:
  %x0 = call i32 @foo(i32 %x, i32 0) nounwind readnone
  %y0 = call i32 @foo(i32 %x, i32 1) nounwind readnone
  br label %if.end

if.else:
  %x1 = call i32 @foo(i32 %y, i32 0) nounwind readnone
  %y1 = call i32 @foo(i32 %y, i32 1) nounwind readnone
  br label %if.end

if.end:
  %xx = phi i32 [ %x0, %if.then ], [ %x1, %if.else ]
  %yy = phi i32 [ %y0, %if.then ], [ %y1, %if.else ]
  %ret = add i32 %xx, %yy
  ret i32 %ret
}

; CHECK-LABEL: test3
; CHECK: select
; CHECK: call
; CHECK: call
; CHECK: add
; CHECK-NOT: br

define i32 @test4(i1 zeroext %flag, i32 %x, i32* %y) {
entry:
  br i1 %flag, label %if.then, label %if.else

if.then:
  %a = add i32 %x, 5
  store i32 %a, i32* %y
  br label %if.end

if.else:
  %b = add i32 %x, 7
  store i32 %b, i32* %y
  br label %if.end

if.end:
  ret i32 1
}

; CHECK-LABEL: test4
; CHECK: select
; CHECK: store
; CHECK-NOT: store

define i32 @test5(i1 zeroext %flag, i32 %x, i32* %y) {
entry:
  br i1 %flag, label %if.then, label %if.else

if.then:
  %a = add i32 %x, 5
  store volatile i32 %a, i32* %y
  br label %if.end

if.else:
  %b = add i32 %x, 7
  store i32 %b, i32* %y
  br label %if.end

if.end:
  ret i32 1
}

; CHECK-LABEL: test5
; CHECK: store volatile
; CHECK: store

define i32 @test6(i1 zeroext %flag, i32 %x, i32* %y) {
entry:
  br i1 %flag, label %if.then, label %if.else

if.then:
  %a = add i32 %x, 5
  store volatile i32 %a, i32* %y
  br label %if.end

if.else:
  %b = add i32 %x, 7
  store volatile i32 %b, i32* %y
  br label %if.end

if.end:
  ret i32 1
}

; CHECK-LABEL: test6
; CHECK: select
; CHECK: store volatile
; CHECK-NOT: store

define i32 @test7(i1 zeroext %flag, i32 %x, i32* %y) {
entry:
  br i1 %flag, label %if.then, label %if.else

if.then:
  %z = load volatile i32, i32* %y
  %a = add i32 %z, 5
  store volatile i32 %a, i32* %y
  br label %if.end

if.else:
  %w = load volatile i32, i32* %y
  %b = add i32 %w, 7
  store volatile i32 %b, i32* %y
  br label %if.end

if.end:
  ret i32 1
}

; CHECK-LABEL: test7
; CHECK-DAG: select
; CHECK-DAG: load volatile
; CHECK: store volatile
; CHECK-NOT: load
; CHECK-NOT: store

; %z and %w are in different blocks. We shouldn't sink the add because
; there may be intervening memory instructions.
define i32 @test8(i1 zeroext %flag, i32 %x, i32* %y) {
entry:
  %z = load volatile i32, i32* %y
  br i1 %flag, label %if.then, label %if.else

if.then:
  %a = add i32 %z, 5
  store volatile i32 %a, i32* %y
  br label %if.end

if.else:
  %w = load volatile i32, i32* %y
  %b = add i32 %w, 7
  store volatile i32 %b, i32* %y
  br label %if.end

if.end:
  ret i32 1
}

; CHECK-LABEL: test8
; CHECK: add
; CHECK: add

; The extra store in %if.then means %z and %w are not equivalent.
define i32 @test9(i1 zeroext %flag, i32 %x, i32* %y, i32* %p) {
entry:
  br i1 %flag, label %if.then, label %if.else

if.then:
  store i32 7, i32* %p
  %z = load volatile i32, i32* %y
  store i32 6, i32* %p
  %a = add i32 %z, 5
  store volatile i32 %a, i32* %y
  br label %if.end

if.else:
  %w = load volatile i32, i32* %y
  %b = add i32 %w, 7
  store volatile i32 %b, i32* %y
  br label %if.end

if.end:
  ret i32 1
}

; CHECK-LABEL: test9
; CHECK: add
; CHECK: add

%struct.anon = type { i32, i32 }

; The GEP indexes a struct type so cannot have a variable last index.
define i32 @test10(i1 zeroext %flag, i32 %x, i32* %y, %struct.anon* %s) {
entry:
  br i1 %flag, label %if.then, label %if.else

if.then:
  %dummy = add i32 %x, 5
  %gepa = getelementptr inbounds %struct.anon, %struct.anon* %s, i32 0, i32 0
  store volatile i32 %x, i32* %gepa
  br label %if.end

if.else:
  %dummy1 = add i32 %x, 6
  %gepb = getelementptr inbounds %struct.anon, %struct.anon* %s, i32 0, i32 1
  store volatile i32 %x, i32* %gepb
  br label %if.end

if.end:
  ret i32 1
}

; CHECK-LABEL: test10
; CHECK: getelementptr
; CHECK: getelementptr
; CHECK: phi
; CHECK: store volatile

; The shufflevector's mask operand cannot be merged in a PHI.
define i32 @test11(i1 zeroext %flag, i32 %w, <2 x i32> %x, <2 x i32> %y) {
entry:
  br i1 %flag, label %if.then, label %if.else

if.then:
  %dummy = add i32 %w, 5
  %sv1 = shufflevector <2 x i32> %x, <2 x i32> %y, <2 x i32> <i32 0, i32 1>
  br label %if.end

if.else:
  %dummy1 = add i32 %w, 6
  %sv2 = shufflevector <2 x i32> %x, <2 x i32> %y, <2 x i32> <i32 1, i32 0>
  br label %if.end

if.end:
  %p = phi <2 x i32> [ %sv1, %if.then ], [ %sv2, %if.else ]
  ret i32 1
}

; CHECK-LABEL: test11
; CHECK: shufflevector
; CHECK: shufflevector

; We can't common an intrinsic!
define i32 @test12(i1 zeroext %flag, i32 %w, i32 %x, i32 %y) {
entry:
  br i1 %flag, label %if.then, label %if.else

if.then:
  %dummy = add i32 %w, 5
  %sv1 = call i32 @llvm.ctlz.i32(i32 %x)
  br label %if.end

if.else:
  %dummy1 = add i32 %w, 6
  %sv2 = call i32 @llvm.cttz.i32(i32 %x)
  br label %if.end

if.end:
  %p = phi i32 [ %sv1, %if.then ], [ %sv2, %if.else ]
  ret i32 1
}

declare i32 @llvm.ctlz.i32(i32 %x) readnone
declare i32 @llvm.cttz.i32(i32 %x) readnone

; CHECK-LABEL: test12
; CHECK: call i32 @llvm.ctlz
; CHECK: call i32 @llvm.cttz

; The TBAA metadata should be properly combined.
define i32 @test13(i1 zeroext %flag, i32 %x, i32* %y) {
entry:
  br i1 %flag, label %if.then, label %if.else

if.then:
  %z = load volatile i32, i32* %y
  %a = add i32 %z, 5
  store volatile i32 %a, i32* %y, !tbaa !3
  br label %if.end

if.else:
  %w = load volatile i32, i32* %y
  %b = add i32 %w, 7
  store volatile i32 %b, i32* %y, !tbaa !4
  br label %if.end

if.end:
  ret i32 1
}

!0 = !{ !"an example type tree" }
!1 = !{ !"int", !0 }
!2 = !{ !"float", !0 }
!3 = !{ !"const float", !2, i64 0 }
!4 = !{ !"special float", !2, i64 1 }

; CHECK-LABEL: test13
; CHECK-DAG: select
; CHECK-DAG: load volatile
; CHECK: store volatile {{.*}}, !tbaa !0
; CHECK-NOT: load
; CHECK-NOT: store

; The call should be commoned.
define i32 @test13a(i1 zeroext %flag, i32 %w, i32 %x, i32 %y) {
entry:
  br i1 %flag, label %if.then, label %if.else

if.then:
  %sv1 = call i32 @bar(i32 %x)
  br label %if.end

if.else:
  %sv2 = call i32 @bar(i32 %y)
  br label %if.end

if.end:
  %p = phi i32 [ %sv1, %if.then ], [ %sv2, %if.else ]
  ret i32 1
}
declare i32 @bar(i32)

; CHECK-LABEL: test13a
; CHECK: %[[x:.*]] = select i1 %flag
; CHECK: call i32 @bar(i32 %[[x]])

; The load should be commoned.
define i32 @test14(i1 zeroext %flag, i32 %w, i32 %x, i32 %y, %struct.anon* %s) {
entry:
  br i1 %flag, label %if.then, label %if.else

if.then:
  %dummy = add i32 %x, 1
  %gepa = getelementptr inbounds %struct.anon, %struct.anon* %s, i32 0, i32 1
  %sv1 = load i32, i32* %gepa
  %cmp1 = icmp eq i32 %sv1, 56
  br label %if.end

if.else:
  %dummy2 = add i32 %x, 4
  %gepb = getelementptr inbounds %struct.anon, %struct.anon* %s, i32 0, i32 1
  %sv2 = load i32, i32* %gepb
  %cmp2 = icmp eq i32 %sv2, 57
  br label %if.end

if.end:
  %p = phi i1 [ %cmp1, %if.then ], [ %cmp2, %if.else ]
  ret i32 1
}

; CHECK-LABEL: test14
; CHECK: getelementptr
; CHECK: load
; CHECK-NOT: load

; The load should be commoned.
define i32 @test15(i1 zeroext %flag, i32 %w, i32 %x, i32 %y, %struct.anon* %s) {
entry:
  br i1 %flag, label %if.then, label %if.else

if.then:
  %dummy = add i32 %x, 1
  %gepa = getelementptr inbounds %struct.anon, %struct.anon* %s, i32 0, i32 0
  %sv1 = load i32, i32* %gepa
  %ext1 = zext i32 %sv1 to i64
  %cmp1 = icmp eq i64 %ext1, 56
  br label %if.end

if.else:
  %dummy2 = add i32 %x, 4
  %gepb = getelementptr inbounds %struct.anon, %struct.anon* %s, i32 0, i32 1
  %sv2 = load i32, i32* %gepb
  %ext2 = zext i32 %sv2 to i64
  %cmp2 = icmp eq i64 %ext2, 57
  br label %if.end

if.end:
  %p = phi i1 [ %cmp1, %if.then ], [ %cmp2, %if.else ]
  ret i32 1
}

; CHECK-LABEL: test15
; CHECK: getelementptr
; CHECK: load
; CHECK-NOT: load

define zeroext i1 @test_crash(i1 zeroext %flag, i32* %i4, i32* %m, i32* %n) {
entry:
  br i1 %flag, label %if.then, label %if.else

if.then:
  %tmp1 = load i32, i32* %i4
  %tmp2 = add i32 %tmp1, -1
  store i32 %tmp2, i32* %i4
  br label %if.end

if.else:
  %tmp3 = load i32, i32* %m
  %tmp4 = load i32, i32* %n
  %tmp5 = add i32 %tmp3, %tmp4
  store i32 %tmp5, i32* %i4
  br label %if.end

if.end:
  ret i1 true
}

; CHECK-LABEL: test_crash
; No checks for test_crash - just ensure it doesn't crash!

define zeroext i1 @test16(i1 zeroext %flag, i1 zeroext %flag2, i32 %blksA, i32 %blksB, i32 %nblks) {

entry:
  br i1 %flag, label %if.then, label %if.else

if.then:
  %cmp = icmp uge i32 %blksA, %nblks
  %frombool1 = zext i1 %cmp to i8
  br label %if.end

if.else:
  br i1 %flag2, label %if.then2, label %if.end

if.then2:
  %add = add i32 %nblks, %blksB
  %cmp2 = icmp ule i32 %add, %blksA
  %frombool3 = zext i1 %cmp2 to i8
  br label %if.end

if.end:
  %obeys.0 = phi i8 [ %frombool1, %if.then ], [ %frombool3, %if.then2 ], [ 0, %if.else ]
  %tobool4 = icmp ne i8 %obeys.0, 0
  ret i1 %tobool4
}

; CHECK-LABEL: test16
; CHECK: zext
; CHECK-NOT: zext

define zeroext i1 @test17(i32 %flag, i32 %blksA, i32 %blksB, i32 %nblks) {
entry:
  switch i32 %flag, label %if.end [
    i32 0, label %if.then
    i32 1, label %if.then2
  ]

if.then:
  %cmp = icmp uge i32 %blksA, %nblks
  %frombool1 = zext i1 %cmp to i8
  br label %if.end

if.then2:
  %add = add i32 %nblks, %blksB
  %cmp2 = icmp ule i32 %add, %blksA
  %frombool3 = zext i1 %cmp2 to i8
  br label %if.end

if.end:
  %obeys.0 = phi i8 [ %frombool1, %if.then ], [ %frombool3, %if.then2 ], [ 0, %entry ]
  %tobool4 = icmp ne i8 %obeys.0, 0
  ret i1 %tobool4
}

; CHECK-LABEL: test17
; CHECK: if.then:
; CHECK-NEXT: icmp uge
; CHECK-NEXT: br label %[[x:.*]]

; CHECK: if.then2:
; CHECK-NEXT: add
; CHECK-NEXT: icmp ule
; CHECK-NEXT: br label %[[x]]

; CHECK: [[x]]:
; CHECK-NEXT: %[[y:.*]] = phi i1 [ %cmp
; CHECK-NEXT: %[[z:.*]] = zext i1 %[[y]]
; CHECK-NEXT: br label %if.end

; CHECK: if.end:
; CHECK-NEXT: phi i8
; CHECK-DAG: [ %[[z]], %[[x]] ]
; CHECK-DAG: [ 0, %entry ]

define zeroext i1 @test18(i32 %flag, i32 %blksA, i32 %blksB, i32 %nblks) {
entry:
  switch i32 %flag, label %if.then3 [
    i32 0, label %if.then
    i32 1, label %if.then2
  ]

if.then:
  %cmp = icmp uge i32 %blksA, %nblks
  %frombool1 = zext i1 %cmp to i8
  br label %if.end

if.then2:
  %add = add i32 %nblks, %blksB
  %cmp2 = icmp ule i32 %add, %blksA
  %frombool3 = zext i1 %cmp2 to i8
  br label %if.end

if.then3:
  %add2 = add i32 %nblks, %blksA
  %cmp3 = icmp ule i32 %add2, %blksA
  %frombool4 = zext i1 %cmp3 to i8
  br label %if.end

if.end:
  %obeys.0 = phi i8 [ %frombool1, %if.then ], [ %frombool3, %if.then2 ], [ %frombool4, %if.then3 ]
  %tobool4 = icmp ne i8 %obeys.0, 0
  ret i1 %tobool4
}

; CHECK-LABEL: test18
; CHECK: if.end:
; CHECK-NEXT: %[[x:.*]] = phi i1
; CHECK-DAG: [ %cmp, %if.then ]
; CHECK-DAG: [ %cmp2, %if.then2 ]
; CHECK-DAG: [ %cmp3, %if.then3 ]
; CHECK-NEXT: zext i1 %[[x]] to i8

define i32 @test_pr30188(i1 zeroext %flag, i32 %x) {
entry:
  %y = alloca i32
  %z = alloca i32
  br i1 %flag, label %if.then, label %if.else

if.then:
  store i32 %x, i32* %y
  br label %if.end

if.else:
  store i32 %x, i32* %z
  br label %if.end

if.end:
  ret i32 1
}

; CHECK-LABEL: test_pr30188
; CHECK-NOT: select
; CHECK: store
; CHECK: store

; CHECK: !0 = !{!1, !1, i64 0}
; CHECK: !1 = !{!"float", !2}
; CHECK: !2 = !{!"an example type tree"}
