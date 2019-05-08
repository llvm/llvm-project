; RUN: llc -verify-machineinstrs -mtriple=powerpc-unknown-linux-gnu < %s | FileCheck %s

; Test case adapted from PR40922.

@a.b = internal global i32 0, align 4

define i32 @a() {
entry:
  %call = tail call i32 bitcast (i32 (...)* @d to i32 ()*)()
  %0 = load i32, i32* @a.b, align 4
  %conv = zext i32 %0 to i64
  %add = add nuw nsw i64 %conv, 6
  %and = and i64 %add, 8589934575
  %cmp = icmp ult i64 %and, %conv
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %call3 = tail call i32 bitcast (i32 (...)* @e to i32 ()*)()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  store i32 %call, i32* @a.b, align 4
  ret i32 undef
}

; CHECK-LABEL: @a
; CHECK: li 5, 0
; CHECK: mr 30, 3
; CHECK: addic 6, 4, 6
; CHECK: addze 5, 5
; CHECK: rlwinm 6, 6, 0, 28, 26
; CHECK: andi. 5, 5, 1

declare i32 @d(...)

declare i32 @e(...)
