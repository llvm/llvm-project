; RUN: llc -verify-machineinstrs %s -mtriple=powerpc64-unknown-linux-gnu -O2 -o - -optimize-regalloc=false -regalloc=fast | FileCheck %s

declare void @func(i8*, i64, i64)

define void @test(i8* %context, i32** %elementArrayPtr, i32 %value) {
entry:
  %cmp = icmp eq i32 %value, 0
  br i1 %cmp, label %lreturn, label %lnext

lnext:
  %elementArray = load i32*, i32** %elementArrayPtr, align 8
; CHECK: lwz [[LDREG:[0-9]+]], 140(1)                   # 4-byte Folded Reload
; CHECK: # implicit-def: $x[[TEMPREG:[0-9]+]]
  %element = load i32, i32* %elementArray, align 4
; CHECK: mr [[TEMPREG]], [[LDREG]]
; CHECK: clrldi   4, [[TEMPREG]], 32
  %element.ext = zext i32 %element to i64
  %value.ext = zext i32 %value to i64
  call void @func(i8* %context, i64 %value.ext, i64 %element.ext)
  br label %lreturn

lreturn:
  ret void
}
