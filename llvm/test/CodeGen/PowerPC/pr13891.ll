; RUN: llc -verify-machineinstrs < %s | FileCheck %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

%struct.foo = type { i8, i8 }

define void @_Z5check3foos(ptr nocapture byval(%struct.foo) %f, i16 signext %i) noinline {
; CHECK-LABEL: _Z5check3foos:
; CHECK: sth 3, {{[0-9]+}}(1)
; CHECK: lbz {{[0-9]+}}, {{[0-9]+}}(1)
entry:
  %0 = load i16, ptr %f, align 2
  %bf.val.sext = ashr i16 %0, 8
  %cmp = icmp eq i16 %bf.val.sext, %i
  br i1 %cmp, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %conv = sext i16 %bf.val.sext to i32
  tail call void @exit(i32 %conv)
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
  ret void
}

declare void @exit(i32)
