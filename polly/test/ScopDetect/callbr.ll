; RUN: opt %loadPolly -polly-detect -polly-detect-track-failures -disable-output -pass-remarks-missed=polly-detect < %s 2>&1 | FileCheck %s --check-prefix=REMARK
; RUN: opt %loadPolly -polly-detect -polly-detect-track-failures -disable-output -stats                            < %s 2>&1 | FileCheck %s --check-prefix=STAT
; REQUIRES: asserts

; REMARK: Branch from indirect terminator.

; STAT: 1 polly-detect - Number of rejected regions: Branch from indirect terminator


target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @func(i32 %n, ptr noalias nonnull %A) {
entry:
  callbr void asm sideeffect "", "!i,~{dirflag},~{fpsr},~{flags}"() #1
          to label %fallthrough [label %for]

fallthrough:
  br label %for

for:
  %j = phi i32 [0, %entry], [0, %fallthrough], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, %n
  br i1 %j.cmp, label %body, label %exit

    body:
      store double 42.0, ptr %A
      br label %inc

inc:
  %j.inc = add nuw nsw i32 %j, 1
  br label %for

exit:
  br label %return

return:
  ret void
}
