; RUN: llc -mtriple=nanomips -verify-machineinstrs < %s | FileCheck %s

declare void @exit(i32)

; CHECK-NOT: MBB exits via unconditional fall-through but ends with a barrier instruction!
define i32 @main() {
  call void @exit(i32 0)
  unreachable
}

