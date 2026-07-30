; RUN: opt -S -aa-pipeline=basic-aa,objc-arc-aa -passes=gvn < %s | FileCheck %s

@x = common global ptr null, align 8

declare ptr @llvm.objc.retain(ptr)
declare i32 @llvm.objc.sync.enter(ptr)
declare i32 @llvm.objc.sync.exit(ptr)

; GVN should be able to eliminate this redundant load, with ARC-specific
; alias analysis.

; CHECK: define ptr @test0(i32 %n)
; CHECK-NEXT: entry:
; CHECK-NEXT: %s = load ptr, ptr @x
; CHECK-NOT: load
; CHECK: ret ptr %s
; CHECK-NEXT: }
define ptr @test0(i32 %n) nounwind {
entry:
  %s = load ptr, ptr @x
  %0 = tail call ptr @llvm.objc.retain(ptr %s) nounwind
  %t = load ptr, ptr @x
  ret ptr %t
}

; GVN should not be able to eliminate this redundant load, with ARC-specific
; alias analysis.

; CHECK-LABEL: define ptr @test1(
; CHECK: load
; CHECK: load
; CHECK: ret ptr %t
; CHECK: }
define ptr @test1(i32 %n) nounwind {
entry:
  %s = load ptr, ptr @x
  %0 = call i32 @llvm.objc.sync.enter(ptr %s)
  %t = load ptr, ptr @x
  %1 = call i32 @llvm.objc.sync.exit(ptr %s)
  ret ptr %t
}
