; Make sure memaccess checks preceed the following reads.
;
; RUN: opt < %s -disable-output 2>&1 -passes='hwasan,print<memoryssa>' -hwasan-use-stack-safety=0 -mtriple aarch64-linux-android30 | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-android10000"

declare void @use32(ptr)

define i32 @test_alloca() sanitize_hwaddress {
entry:
  %x = alloca i32, align 4
  ; CHECK: call void @use32
  call void @use32(ptr nonnull %x)
  ; CHECK: [[A:[0-9]+]] = MemoryDef({{[0-9]+}})
  ; CHECK-NEXT: call void @llvm.hwasan.check.memaccess.shortgranules
  ; CHECK: MemoryUse([[A]])
  ; CHECK-NEXT: load i32, ptr %x.hwasan
  %y = load i32, ptr %x
  ; CHECK: {{[0-9]+}} = MemoryDef([[A]])
  ; CHECK-NEXT: call void @llvm.memset.p0.i64
  ret i32 %y
}
