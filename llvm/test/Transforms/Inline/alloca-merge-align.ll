; RUN: opt < %s -inline -S | FileCheck %s

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

%struct.s = type { i32, i32 }

define void @foo(%struct.s* byval nocapture readonly %a) {
entry:
  %x = alloca [2 x i32], align 4
  %a1 = getelementptr inbounds %struct.s, %struct.s* %a, i64 0, i32 0
  %0 = load i32, i32* %a1, align 4
  %arrayidx = getelementptr inbounds [2 x i32], [2 x i32]* %x, i64 0, i64 0
  store i32 %0, i32* %arrayidx, align 4
  %b = getelementptr inbounds %struct.s, %struct.s* %a, i64 0, i32 1
  %1 = load i32, i32* %b, align 4
  %arrayidx2 = getelementptr inbounds [2 x i32], [2 x i32]* %x, i64 0, i64 1
  store i32 %1, i32* %arrayidx2, align 4
  call void @bar(i32* %arrayidx) #2
  ret void
}

define void @foo0(%struct.s* byval nocapture readonly %a) {
entry:
  %x = alloca [2 x i32]
  %a1 = getelementptr inbounds %struct.s, %struct.s* %a, i64 0, i32 0
  %0 = load i32, i32* %a1, align 4
  %arrayidx = getelementptr inbounds [2 x i32], [2 x i32]* %x, i64 0, i64 0
  store i32 %0, i32* %arrayidx, align 4
  %b = getelementptr inbounds %struct.s, %struct.s* %a, i64 0, i32 1
  %1 = load i32, i32* %b, align 4
  %arrayidx2 = getelementptr inbounds [2 x i32], [2 x i32]* %x, i64 0, i64 1
  store i32 %1, i32* %arrayidx2, align 4
  call void @bar(i32* %arrayidx) #2
  ret void
}

define void @foo1(%struct.s* byval nocapture readonly %a) {
entry:
  %x = alloca [2 x i32], align 1
  %a1 = getelementptr inbounds %struct.s, %struct.s* %a, i64 0, i32 0
  %0 = load i32, i32* %a1, align 4
  %arrayidx = getelementptr inbounds [2 x i32], [2 x i32]* %x, i64 0, i64 0
  store i32 %0, i32* %arrayidx, align 4
  %b = getelementptr inbounds %struct.s, %struct.s* %a, i64 0, i32 1
  %1 = load i32, i32* %b, align 4
  %arrayidx2 = getelementptr inbounds [2 x i32], [2 x i32]* %x, i64 0, i64 1
  store i32 %1, i32* %arrayidx2, align 4
  call void @bar(i32* %arrayidx) #2
  ret void
}

declare void @bar(i32*) #1

define void @goo(%struct.s* byval nocapture readonly %a) {
entry:
  %x = alloca [2 x i32], align 32
  %a1 = getelementptr inbounds %struct.s, %struct.s* %a, i64 0, i32 0
  %0 = load i32, i32* %a1, align 4
  %arrayidx = getelementptr inbounds [2 x i32], [2 x i32]* %x, i64 0, i64 0
  store i32 %0, i32* %arrayidx, align 32
  %b = getelementptr inbounds %struct.s, %struct.s* %a, i64 0, i32 1
  %1 = load i32, i32* %b, align 4
  %arrayidx2 = getelementptr inbounds [2 x i32], [2 x i32]* %x, i64 0, i64 1
  store i32 %1, i32* %arrayidx2, align 4
  call void @bar(i32* %arrayidx) #2
  ret void
}

; CHECK-LABEL: @main
; CHECK: alloca [2 x i32], align 32
; CHECK-NOT: alloca [2 x i32]
; CHECK: ret i32 0

define signext i32 @main() {
entry:
  %a = alloca i64, align 8
  %tmpcast = bitcast i64* %a to %struct.s*
  store i64 0, i64* %a, align 8
  %a1 = bitcast i64* %a to i32*
  store i32 1, i32* %a1, align 8
  call void @foo(%struct.s* byval %tmpcast)
  store i32 2, i32* %a1, align 8
  call void @goo(%struct.s* byval %tmpcast)
  ret i32 0
}

; CHECK-LABEL: @test0
; CHECK: alloca [2 x i32], align 32
; CHECK-NOT: alloca [2 x i32]
; CHECK: ret i32 0

define signext i32 @test0() {
entry:
  %a = alloca i64, align 8
  %tmpcast = bitcast i64* %a to %struct.s*
  store i64 0, i64* %a, align 8
  %a1 = bitcast i64* %a to i32*
  store i32 1, i32* %a1, align 8
  call void @foo0(%struct.s* byval %tmpcast)
  store i32 2, i32* %a1, align 8
  call void @goo(%struct.s* byval %tmpcast)
  ret i32 0
}
