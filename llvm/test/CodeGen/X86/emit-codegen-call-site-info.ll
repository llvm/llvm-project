; RUN: llc -mtriple=x86_64-linux-gnu -emit-codegen-call-site-info %s -o - -stop-before=finalize-isel | FileCheck %s --check-prefix=ON
; RUN: llc -mtriple=x86_64-linux-gnu -emit-codegen-call-site-info=false %s -o - -stop-before=finalize-isel | FileCheck %s --check-prefix=OFF

; ON: callSites:
; ON-NEXT:   - {
; OFF: callSites:       []

define i32 @caller(i32 %a, i32 %b, i32 %c) {
entry:
  %add = add nsw i32 %b, %a
  %call = tail call i32 @callee(i32 %add, i32 %c, i32 10)
  ret i32 %call
}

declare i32 @callee(i32, i32, i32)
