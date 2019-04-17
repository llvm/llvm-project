; RUN: opt < %s -codegenprepare -S -mtriple=x86_64-unknown-unknown    | FileCheck %s


@a = global [10 x i8] zeroinitializer, align 1
declare void @foo()

; ext(and(ld, cst)) -> and(ext(ld), ext(cst))
define void @test1(i32* %p, i32 %ll) {
; CHECK-LABEL: @test1
; CHECK-NEXT:  entry:
; CHECK-NEXT:    load
; CHECK-NEXT:    zext
; CHECK-NEXT:    and
entry:
  %tmp = load i8, i8* getelementptr inbounds ([10 x i8], [10 x i8]* @a, i64 0, i64 0), align 1
  %and = and i8 %tmp, 60
  %cmp = icmp ugt i8 %and, 20
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %conv2 = zext i8 %and to i32
  %add = add nsw i32 %conv2, %ll
  store i32 %add, i32* %p, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  tail call void @foo()
  ret void
}

; ext(or(ld, cst)) -> or(ext(ld), ext(cst))
define void @test2(i32* %p, i32 %ll) {
; CHECK-LABEL: @test2
; CHECK-NEXT:  entry:
; CHECK-NEXT:    load
; CHECK-NEXT:    zext
; CHECK-NEXT:    or
entry:
  %tmp = load i8, i8* getelementptr inbounds ([10 x i8], [10 x i8]* @a, i64 0, i64 0), align 1
  %or = or i8 %tmp, 60
  %cmp = icmp ugt i8 %or, 20
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %conv2 = zext i8 %or to i32
  %add = add nsw i32 %conv2, %ll
  store i32 %add, i32* %p, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  tail call void @foo()
  ret void
}

; ext(and(shl(ld, cst), cst)) -> and(shl(ext(ld), ext(cst)), ext(cst))
define void @test3(i32* %p, i32 %ll) {
; CHECK-LABEL: @test3
; CHECK-NEXT:  entry:
; CHECK-NEXT:    load
; CHECK-NEXT:    zext
; CHECK-NEXT:    shl
; CHECK-NEXT:    and
entry:
  %tmp = load i8, i8* getelementptr inbounds ([10 x i8], [10 x i8]* @a, i64 0, i64 0), align 1
  %shl = shl i8 %tmp, 2
  %and = and i8 %shl, 60
  %cmp = icmp ugt i8 %and, 20
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %conv2 = zext i8 %and to i32
  %add = add nsw i32 %conv2, %ll
  store i32 %add, i32* %p, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  tail call void @foo()
  ret void
}

; zext(shrl(ld, cst)) -> shrl(zext(ld), zext(cst))
define void @test4(i32* %p, i32 %ll) {
; CHECK-LABEL: @test4
; CHECK-NEXT:  entry:
; CHECK-NEXT:    load
; CHECK-NEXT:    zext
; CHECK-NEXT:    lshr
entry:
  %tmp = load i8, i8* getelementptr inbounds ([10 x i8], [10 x i8]* @a, i64 0, i64 0), align 1
  %lshr = lshr i8 %tmp, 2
  %cmp = icmp ugt i8 %lshr, 20
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %conv2 = zext i8 %lshr to i32
  %add = add nsw i32 %conv2, %ll
  store i32 %add, i32* %p, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  tail call void @foo()
  ret void
}

; ext(xor(ld, cst)) -> xor(ext(ld), ext(cst))
define void @test5(i32* %p, i32 %ll) {
; CHECK-LABEL: @test5
; CHECK-NEXT:  entry:
; CHECK-NEXT:    load
; CHECK-NEXT:    zext
; CHECK-NEXT:    xor
entry:
  %tmp = load i8, i8* getelementptr inbounds ([10 x i8], [10 x i8]* @a, i64 0, i64 0), align 1
  %xor = xor i8 %tmp, 60
  %cmp = icmp ugt i8 %xor, 20
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %conv2 = zext i8 %xor to i32
  %add = add nsw i32 %conv2, %ll
  store i32 %add, i32* %p, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  tail call void @foo()
  ret void
}

