; RUN: not llc < %s 2>&1 | FileCheck %s
; RUN: not llc -fast-isel=true < %s 2>&1 | FileCheck %s
; RUN: not llc -global-isel=true -global-isel-abort=2 < %s 2>&1 | FileCheck %s

; CHECK: LLVM ERROR: Calls that require saving ZA non-lazily is not yet implemented

target triple = "aarch64"

define void @foo(ptr %f) #0 {
  call void %f() "aarch64_inout_zt0"
  ret void
}

attributes #0 = { "aarch64_inout_zt0" "aarch64_inout_za" "target-feature"="+sme,+sve" }
