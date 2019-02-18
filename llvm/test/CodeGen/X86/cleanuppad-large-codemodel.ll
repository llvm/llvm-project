; RUN: llc -mtriple=x86_64-pc-windows-msvc -code-model=large -relocation-model=static -o - < %s | FileCheck %s

declare i32 @__CxxFrameHandler3(...)

declare void @bar()

define void @foo() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  invoke void @bar()
    to label %exit unwind label %cleanup
cleanup:
  %c = cleanuppad within none []
  call void @bar() [ "funclet"(token %c) ]
  cleanupret from %c unwind to caller
exit:
  ret void
}

; CHECK: foo: # @foo
; CHECK: movabsq $bar, %[[reg:[^ ]*]]
; CHECK: callq *%[[reg]]
; CHECK: retq

; CHECK: "?dtor$2@?0?foo@4HA":
; CHECK: movabsq $bar, %[[reg:[^ ]*]]
; CHECK: callq *%[[reg]]
; CHECK: retq                            # CLEANUPRET
