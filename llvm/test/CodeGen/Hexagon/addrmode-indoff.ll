; RUN: llc -march=hexagon < %s | FileCheck %s
;
; Bug 6840. Use absolute+index addressing.

@ga = common global [1024 x i8] zeroinitializer, align 8

; CHECK-LABEL: test0
; CHECK: memub(r{{[0-9]+}}+##ga)
define zeroext i8 @test0(i32 %i) nounwind readonly {
entry:
  %t = getelementptr inbounds [1024 x i8], [1024 x i8]* @ga, i32 0, i32 %i
  %0 = load i8, i8* %t, align 1
  ret i8 %0
}

; CHECK-LABEL: test1
; CHECK: memb(r{{[0-9]+}}+##ga)
define signext i8 @test1(i32 %i) nounwind readonly {
entry:
  %t = getelementptr inbounds [1024 x i8], [1024 x i8]* @ga, i32 0, i32 %i
  %0 = load i8, i8* %t, align 1
  ret i8 %0
}

; CHECK-LABEL: test2
; CHECK: memub(r{{[0-9]+}}<<#1+##ga)
define zeroext i8 @test2(i32 %i) nounwind readonly {
entry:
  %j = shl nsw i32 %i, 1
  %t = getelementptr inbounds [1024 x i8], [1024 x i8]* @ga, i32 0, i32 %j
  %0 = load i8, i8* %t, align 1
  ret i8 %0
}

; CHECK-LABEL: test3
; CHECK: memb(r{{[0-9]+}}<<#1+##ga)
define signext i8 @test3(i32 %i) nounwind readonly {
entry:
  %j = shl nsw i32 %i, 1
  %t = getelementptr inbounds [1024 x i8], [1024 x i8]* @ga, i32 0, i32 %j
  %0 = load i8, i8* %t, align 1
  ret i8 %0
}

; CHECK-LABEL: test4
; CHECK: memub(r{{[0-9]+}}<<#2+##ga)
define zeroext i8 @test4(i32 %i) nounwind readonly {
entry:
  %j = shl nsw i32 %i, 2
  %t = getelementptr inbounds [1024 x i8], [1024 x i8]* @ga, i32 0, i32 %j
  %0 = load i8, i8* %t, align 1
  ret i8 %0
}

; CHECK-LABEL: test5
; CHECK: memb(r{{[0-9]+}}<<#2+##ga)
define signext i8 @test5(i32 %i) nounwind readonly {
entry:
  %j = shl nsw i32 %i, 2
  %t = getelementptr inbounds [1024 x i8], [1024 x i8]* @ga, i32 0, i32 %j
  %0 = load i8, i8* %t, align 1
  ret i8 %0
}

; CHECK-LABEL: test10
; CHECK: memb(r{{[0-9]+}}+##ga)
define void @test10(i32 %i, i8 zeroext %v) nounwind {
entry:
  %t = getelementptr inbounds [1024 x i8], [1024 x i8]* @ga, i32 0, i32 %i
  store i8 %v, i8* %t, align 1
  ret void
}

; CHECK-LABEL: test11
; CHECK: memb(r{{[0-9]+}}<<#1+##ga)
define void @test11(i32 %i, i8 signext %v) nounwind {
entry:
  %j = shl nsw i32 %i, 1
  %t = getelementptr inbounds [1024 x i8], [1024 x i8]* @ga, i32 0, i32 %j
  store i8 %v, i8* %t, align 1
  ret void
}

; CHECK-LABEL: test12
; CHECK: memb(r{{[0-9]+}}<<#2+##ga)
define void @test12(i32 %i, i8 zeroext %v) nounwind {
entry:
  %j = shl nsw i32 %i, 2
  %t = getelementptr inbounds [1024 x i8], [1024 x i8]* @ga, i32 0, i32 %j
  store i8 %v, i8* %t, align 1
  ret void
}
