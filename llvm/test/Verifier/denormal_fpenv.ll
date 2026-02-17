; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

declare void @func()

; CHECK: denormal_fpenv attribute may not apply to call sites
; CHECK-NEXT: call void @func() #0
define void @no_callsites() {
  call void @func() denormal_fpenv(preservesign)
  ret void
}
