; RUN: opt -aa-pipeline=basic-aa -passes='print<memoryssa>,verify<memoryssa>' -disable-output < %s 2>&1 | FileCheck %s --implicit-check-not=MemoryDef
;
; Ensures that allow.*.check are treated as not reading or writing memory.

target triple = "aarch64-linux"

define i1 @test_runtime(ptr %a) local_unnamed_addr {
entry:
; CHECK: 1 = MemoryDef(liveOnEntry)
  store i32 4, ptr %a, align 4
  %allow = call i1 @llvm.allow.runtime.check(metadata !"test_check")
  %0 = load i32, ptr %a, align 4
; CHECK: MemoryUse(1)
  ret i1 %allow
}

declare i1 @llvm.allow.runtime.check(metadata)

define i1 @test_ubsan(ptr %a) local_unnamed_addr {
entry:
; CHECK: 1 = MemoryDef(liveOnEntry)
  store i32 4, ptr %a, align 4
  %allow = call i1 @llvm.allow.ubsan.check(i8 7)
  %0 = load i32, ptr %a, align 4
; CHECK: MemoryUse(1)
  ret i1 %allow
}

declare i1 @llvm.allow.ubsan.check(i8)
