; RUN: opt -S -aa-pipeline=basic-aa,objc-arc-aa -passes=gvn < %s | FileCheck %s

@x = common global i8* null, align 8

declare i8* @llvm.objc.retain(i8*)
declare i32 @llvm.objc.sync.enter(i8*)
declare i32 @llvm.objc.sync.exit(i8*)

; GVN should be able to eliminate this redundant load, with ARC-specific
; alias analysis.

; CHECK: define i8* @test0(i32 %n)
; CHECK-NEXT: entry:
; CHECK-NEXT: %s = load i8*, i8** @x
; CHECK-NOT: load
; CHECK: ret i8* %s
; CHECK-NEXT: }
define i8* @test0(i32 %n) nounwind {
entry:
  %s = load i8*, i8** @x
  %0 = tail call i8* @llvm.objc.retain(i8* %s) nounwind
  %t = load i8*, i8** @x
  ret i8* %t
}

; GVN should not be able to eliminate this redundant load, with ARC-specific
; alias analysis.

; CHECK-LABEL: define i8* @test1(
; CHECK: load
; CHECK: load
; CHECK: ret i8* %t
; CHECK: }
define i8* @test1(i32 %n) nounwind {
entry:
  %s = load i8*, i8** @x
  %0 = call i32 @llvm.objc.sync.enter(i8* %s)
  %t = load i8*, i8** @x
  %1 = call i32 @llvm.objc.sync.exit(i8* %s)
  ret i8* %t
}
