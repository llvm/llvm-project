; RUN: opt -passes=lowertypetests -lowertypetests-summary-action=import -lowertypetests-read-summary=%p/Inputs/blockaddr-import.yaml %s -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux"

declare i1 @llvm.type.test(i8*, metadata) #1
declare !type !11 i32 @o(...)

define hidden void @m() #0 !type !3 {
entry:
  br label %n
n:
  %call = tail call i32 (i8*, ...) bitcast (i32 (...)* @o to i32 (i8*, ...)*)(i8* blockaddress(@m, %n)) #4
; Make sure that blockaddress refers to the new function, m.cfi
; CHECK: define hidden void @m.cfi()
; CHECK: blockaddress(@m.cfi, %n)

  ret void
}

!3 = !{i64 0, !"_ZTSFvE"}
!11 = !{i64 0, !"_ZTSFiE"}
