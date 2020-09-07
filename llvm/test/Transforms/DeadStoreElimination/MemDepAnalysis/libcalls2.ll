; RUN: opt -S -basic-aa -dse -enable-dse-memoryssa=false < %s | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

declare i8* @strncpy(i8* %dest, i8* %src, i32 %n) nounwind
define void @test2(i8* %src) {
; CHECK-LABEL: @test2(
  %B = alloca [16 x i8]
  %dest = getelementptr inbounds [16 x i8], [16 x i8]* %B, i64 0, i64 0
; CHECK: @strncpy
  %call = call i8* @strncpy(i8* %dest, i8* %src, i32 12)
; CHECK: ret void
  ret void
}
