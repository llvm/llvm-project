; RUN: llc -mtriple=arm64ec-unknown-windows < %s | FileCheck -check-prefixes=CHECK,NONGNU %s
; RUN: llc -mtriple=arm64ec-unknown-windows-gnu < %s | FileCheck -check-prefixes=CHECK,GNU %s

; CHECK-LABEL: func = "#func"
; CHECK: bl "#other"
; NONGNU: bl "#__security_check_cookie_arm64ec"
; GNU: bl "#__stack_chk_fail"
define void @func() #0 {
entry:
  %buf = alloca [10 x i8], align 1
  call void @other(ptr %buf) #1
  ret void
}

declare void @other(ptr) #1

attributes #0 = { nounwind sspstrong }
attributes #1 = { nounwind }
