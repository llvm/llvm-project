; RUN: llc -O3 -o - %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: uwtable
; When tail-duplicating during placement, we work backward from blocks with
; multiple successors. In this case, the block dup1 gets duplicated into dup2
; and if.then64, and then the block dup2 gets duplicated into land.lhs.true
; and if.end70
; CHECK-LABEL: repeated_tail_dup:
define void @repeated_tail_dup(i1 %a1, i1 %a2, i32* %a4, i32* %a5, i8* %a6) #0 align 2 {
entry:
  br label %for.cond

; CHECK: {{^}}.[[HEADER:LBB0_[1-9]]]: # %for.cond
for.cond:                                         ; preds = %dup1, %entry
  br i1 %a1, label %land.lhs.true, label %if.end56

land.lhs.true:                                    ; preds = %for.cond
  store i32 10, i32* %a4, align 8
  br label %dup2

if.end56:                                         ; preds = %for.cond
  br i1 %a2, label %if.then64, label %if.end70

if.then64:                                        ; preds = %if.end56
  store i8 1, i8* %a6, align 1
  br label %dup1

; CHECK:      # %if.end70
; CHECK-NEXT: # in Loop:
; CHECK-NEXT: movl $12, (%rdx)
; CHECK-NEXT: movl $2, (%rcx)
; CHECK-NEXT: testl %eax, %eax
; CHECK-NEXT: je .[[HEADER]]
if.end70:                                         ; preds = %if.end56
  store i32 12, i32* %a4, align 8
  br label %dup2

dup2:                                             ; preds = %if.end70, %land.lhs.true
  store i32 2, i32* %a5, align 4
  br label %dup1

dup1:                                             ; preds = %dup2, %if.then64
  %val = load i32, i32* %a4, align 8
  %switch = icmp ult i32 undef, 1
  br i1 %switch, label %for.cond, label %for.end

for.end:                                          ; preds = %dup1
  ret void
}

attributes #0 = { uwtable }
