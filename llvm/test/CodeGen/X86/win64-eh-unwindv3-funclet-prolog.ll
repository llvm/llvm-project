; RUN: llc -mtriple=x86_64-unknown-windows-msvc -o - %s | FileCheck %s

; Test that the V3 pass correctly handles funclets. Each funclet gets its own
; UNWIND_INFO, so V3 capacity limits (≤31 prolog ops, ≤7 epilogs) apply per
; funclet, not summed across the entire function.
;
; This function has 15 cleanup funclets, each with its own prolog/epilog.
; Each funclet generates ~2 prolog ops (push rbp + stack alloc), plus the
; main function has ~3 (push rbp + stack alloc + set frame). The total across
; all funclets (~33) exceeds the 31-op V3 limit, but each individual funclet
; is well within limits. The V3 pass must not report_fatal_error for this case.

declare i32 @c(i32) local_unnamed_addr
declare void @cleanup_helper(i32) local_unnamed_addr

; CHECK-LABEL: many_funclets:
; CHECK:       .seh_proc many_funclets
; CHECK:       .seh_endprologue
; CHECK-NOT:   .seh_splitchained
; CHECK:       .seh_endproc

define dso_local i32 @many_funclets(i32 %x) local_unnamed_addr personality ptr @__C_specific_handler {
entry:
  %v0 = invoke i32 @c(i32 %x)
    to label %try1 unwind label %cleanup0

try1:
  %v1 = invoke i32 @c(i32 %v0)
    to label %try2 unwind label %cleanup1

try2:
  %v2 = invoke i32 @c(i32 %v1)
    to label %try3 unwind label %cleanup2

try3:
  %v3 = invoke i32 @c(i32 %v2)
    to label %try4 unwind label %cleanup3

try4:
  %v4 = invoke i32 @c(i32 %v3)
    to label %try5 unwind label %cleanup4

try5:
  %v5 = invoke i32 @c(i32 %v4)
    to label %try6 unwind label %cleanup5

try6:
  %v6 = invoke i32 @c(i32 %v5)
    to label %try7 unwind label %cleanup6

try7:
  %v7 = invoke i32 @c(i32 %v6)
    to label %try8 unwind label %cleanup7

try8:
  %v8 = invoke i32 @c(i32 %v7)
    to label %try9 unwind label %cleanup8

try9:
  %v9 = invoke i32 @c(i32 %v8)
    to label %try10 unwind label %cleanup9

try10:
  %v10 = invoke i32 @c(i32 %v9)
    to label %try11 unwind label %cleanup10

try11:
  %v11 = invoke i32 @c(i32 %v10)
    to label %try12 unwind label %cleanup11

try12:
  %v12 = invoke i32 @c(i32 %v11)
    to label %try13 unwind label %cleanup12

try13:
  %v13 = invoke i32 @c(i32 %v12)
    to label %try14 unwind label %cleanup13

try14:
  %v14 = invoke i32 @c(i32 %v13)
    to label %done unwind label %cleanup14

done:
  ret i32 %v14

cleanup0:
  %tok0 = cleanuppad within none []
  call void @cleanup_helper(i32 0) [ "funclet"(token %tok0) ]
  cleanupret from %tok0 unwind to caller

cleanup1:
  %tok1 = cleanuppad within none []
  call void @cleanup_helper(i32 1) [ "funclet"(token %tok1) ]
  cleanupret from %tok1 unwind to caller

cleanup2:
  %tok2 = cleanuppad within none []
  call void @cleanup_helper(i32 2) [ "funclet"(token %tok2) ]
  cleanupret from %tok2 unwind to caller

cleanup3:
  %tok3 = cleanuppad within none []
  call void @cleanup_helper(i32 3) [ "funclet"(token %tok3) ]
  cleanupret from %tok3 unwind to caller

cleanup4:
  %tok4 = cleanuppad within none []
  call void @cleanup_helper(i32 4) [ "funclet"(token %tok4) ]
  cleanupret from %tok4 unwind to caller

cleanup5:
  %tok5 = cleanuppad within none []
  call void @cleanup_helper(i32 5) [ "funclet"(token %tok5) ]
  cleanupret from %tok5 unwind to caller

cleanup6:
  %tok6 = cleanuppad within none []
  call void @cleanup_helper(i32 6) [ "funclet"(token %tok6) ]
  cleanupret from %tok6 unwind to caller

cleanup7:
  %tok7 = cleanuppad within none []
  call void @cleanup_helper(i32 7) [ "funclet"(token %tok7) ]
  cleanupret from %tok7 unwind to caller

cleanup8:
  %tok8 = cleanuppad within none []
  call void @cleanup_helper(i32 8) [ "funclet"(token %tok8) ]
  cleanupret from %tok8 unwind to caller

cleanup9:
  %tok9 = cleanuppad within none []
  call void @cleanup_helper(i32 9) [ "funclet"(token %tok9) ]
  cleanupret from %tok9 unwind to caller

cleanup10:
  %tok10 = cleanuppad within none []
  call void @cleanup_helper(i32 10) [ "funclet"(token %tok10) ]
  cleanupret from %tok10 unwind to caller

cleanup11:
  %tok11 = cleanuppad within none []
  call void @cleanup_helper(i32 11) [ "funclet"(token %tok11) ]
  cleanupret from %tok11 unwind to caller

cleanup12:
  %tok12 = cleanuppad within none []
  call void @cleanup_helper(i32 12) [ "funclet"(token %tok12) ]
  cleanupret from %tok12 unwind to caller

cleanup13:
  %tok13 = cleanuppad within none []
  call void @cleanup_helper(i32 13) [ "funclet"(token %tok13) ]
  cleanupret from %tok13 unwind to caller

cleanup14:
  %tok14 = cleanuppad within none []
  call void @cleanup_helper(i32 14) [ "funclet"(token %tok14) ]
  cleanupret from %tok14 unwind to caller
}

declare i32 @__C_specific_handler(...)

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"winx64-eh-unwind", i32 3}
