; RUN: llc -mtriple thumbv7-apple-ios3.0.0 -warn-stack-size=80 < %s 2>&1 >/dev/null | FileCheck %s
; Check the internal option that warns when the stack size exceeds the
; given amount.
; <rdar://13987214>

; CHECK-NOT: nowarn
define void @nowarn() nounwind ssp "frame-pointer"="all" {
entry:
  %buffer = alloca [12 x i8], align 1
  %arraydecay = getelementptr inbounds [12 x i8], [12 x i8]* %buffer, i64 0, i64 0
  call void @doit(i8* %arraydecay) nounwind
  ret void
}

; CHECK: warning: stack size limit exceeded (92) in warn
define void @warn() nounwind ssp "frame-pointer"="all" {
entry:
  %buffer = alloca [80 x i8], align 1
  %arraydecay = getelementptr inbounds [80 x i8], [80 x i8]* %buffer, i64 0, i64 0
  call void @doit(i8* %arraydecay) nounwind
  ret void
}

declare void @doit(i8*)
