; RUN: opt -codegenprepare -S < %s | FileCheck %s
; RUN: opt -enable-debugify -codegenprepare -S < %s 2>&1 | FileCheck %s -check-prefix=DEBUG

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"

; CHECK-LABEL: @test1(
; CHECK: llvm.uadd.with.overflow
; CHECK: ret i64
define i64 @test1(i64 %a, i64 %b) nounwind ssp {
entry:
  %add = add i64 %b, %a
  %cmp = icmp ult i64 %add, %a
  %Q = select i1 %cmp, i64 %b, i64 42
  ret i64 %Q
}

; CHECK-LABEL: @test2(
; CHECK: llvm.uadd.with.overflow
; CHECK: ret i64
define i64 @test2(i64 %a, i64 %b) nounwind ssp {
entry:
  %add = add i64 %b, %a
  %cmp = icmp ult i64 %add, %b
  %Q = select i1 %cmp, i64 %b, i64 42
  ret i64 %Q
}

; CHECK-LABEL: @test3(
; CHECK: llvm.uadd.with.overflow
; CHECK: ret i64
define i64 @test3(i64 %a, i64 %b) nounwind ssp {
entry:
  %add = add i64 %b, %a
  %cmp = icmp ugt i64 %b, %add
  %Q = select i1 %cmp, i64 %b, i64 42
  ret i64 %Q
}

; CHECK-LABEL: @test4(
; CHECK: llvm.uadd.with.overflow
; CHECK: extractvalue
; CHECK: extractvalue
; CHECK: select
define i64 @test4(i64 %a, i64 %b, i1 %c) nounwind ssp {
entry:
  %add = add i64 %b, %a
  %cmp = icmp ugt i64 %b, %add
  br i1 %c, label %next, label %exit

 next:
  %Q = select i1 %cmp, i64 %b, i64 42
  ret i64 %Q

 exit:
  ret i64 0
}

; CHECK-LABEL: @test5(
; CHECK-NOT: llvm.uadd.with.overflow
; CHECK: next
define i64 @test5(i64 %a, i64 %b, i64* %ptr, i1 %c) nounwind ssp {
entry:
  %add = add i64 %b, %a
  store i64 %add, i64* %ptr
  %cmp = icmp ugt i64 %b, %add
  br i1 %c, label %next, label %exit

 next:
  %Q = select i1 %cmp, i64 %b, i64 42
  ret i64 %Q

 exit:
  ret i64 0
}

; Check that every instruction inserted by -codegenprepare has a debug location.
; DEBUG: CheckModuleDebugify: PASS
