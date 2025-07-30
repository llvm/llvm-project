; RUN: llc -verify-machineinstrs %s -mtriple=powerpc64-unknown-linux-gnu -O2 -o - -optimize-regalloc=false -regalloc=fast | FileCheck %s

declare void @func(ptr, i64, i64)

define void @test(ptr %context, ptr %elementArrayPtr, i32 %value) {
entry:
  %cmp = icmp eq i32 %value, 0
  br i1 %cmp, label %lreturn, label %lnext

lnext:
  %elementArray = load ptr, ptr %elementArrayPtr, align 8
; CHECK: ld [[LDREG:[0-9]+]], 120(1)                   # 8-byte Folded Reload
; CHECK: # implicit-def: $x[[TEMPREG:[0-9]+]]
  %element = load i32, ptr %elementArray, align 4
; CHECK: mr [[TEMPREG]], [[LDREG]]
; CHECK: clrldi   4, [[TEMPREG]], 32
  %element.ext = zext i32 %element to i64
  %value.ext = zext i32 %value to i64
  call void @func(ptr %context, i64 %value.ext, i64 %element.ext)
  br label %lreturn

lreturn:
  ret void
}
