; RUN: opt -S -basic-aa -dse < %s | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

declare i8* @strcpy(i8* %dest, i8* %src) nounwind
define void @test1(i8* %src) {
; CHECK-LABEL: @test1(
  %B = alloca [16 x i8]
  %dest = getelementptr inbounds [16 x i8], [16 x i8]* %B, i64 0, i64 0
; CHECK-NOT: @strcpy
  %call = call i8* @strcpy(i8* %dest, i8* %src)
; CHECK: ret void
  ret void
}

declare i8* @strncpy(i8* %dest, i8* %src, i64 %n) nounwind
define void @test2(i8* %src) {
; CHECK-LABEL: @test2(
  %B = alloca [16 x i8]
  %dest = getelementptr inbounds [16 x i8], [16 x i8]* %B, i64 0, i64 0
; CHECK-NOT: @strncpy
  %call = call i8* @strncpy(i8* %dest, i8* %src, i64 12)
; CHECK: ret void
  ret void
}

declare i8* @strcat(i8* %dest, i8* %src) nounwind
define void @test3(i8* %src) {
; CHECK-LABEL: @test3(
  %B = alloca [16 x i8]
  %dest = getelementptr inbounds [16 x i8], [16 x i8]* %B, i64 0, i64 0
; CHECK-NOT: @strcat
  %call = call i8* @strcat(i8* %dest, i8* %src)
; CHECK: ret void
  ret void
}

declare i8* @strncat(i8* %dest, i8* %src, i64 %n) nounwind
define void @test4(i8* %src) {
; CHECK-LABEL: @test4(
  %B = alloca [16 x i8]
  %dest = getelementptr inbounds [16 x i8], [16 x i8]* %B, i64 0, i64 0
; CHECK-NOT: @strncat
  %call = call i8* @strncat(i8* %dest, i8* %src, i64 12)
; CHECK: ret void
  ret void
}

define void @test5(i8* nocapture %src) {
; CHECK-LABEL: @test5(
  %dest = alloca [100 x i8], align 16
  %arraydecay = getelementptr inbounds [100 x i8], [100 x i8]* %dest, i64 0, i64 0
  %call = call i8* @strcpy(i8* %arraydecay, i8* %src)
; CHECK: %call = call i8* @strcpy
  %arrayidx = getelementptr inbounds i8, i8* %call, i64 10
  store i8 97, i8* %arrayidx, align 1
  ret void
}

declare void @user(i8* %p)
define void @test6(i8* %src) {
; CHECK-LABEL: @test6(
  %B = alloca [16 x i8]
  %dest = getelementptr inbounds [16 x i8], [16 x i8]* %B, i64 0, i64 0
; CHECK: @strcpy
  %call = call i8* @strcpy(i8* %dest, i8* %src)
; CHECK: @user
  call void @user(i8* %dest)
; CHECK: ret void
  ret void
}

