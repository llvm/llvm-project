; RUN: llc -verify-machineinstrs < %s | FileCheck %s

; BranchFolding used to remove our empty landingpad block, which is
; undesirable.

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc18.0.0"

declare i32 @__C_specific_handler(...)

declare void @bar()

define void @foo(i1 %cond) personality i32 (...)* @__C_specific_handler {
entry:
  br i1 %cond, label %return, label %try

try:                                              ; preds = %entry
  invoke void @bar()
          to label %fallthrough unwind label %dispatch

dispatch:                                         ; preds = %try
  %cs1 = catchswitch within none [label %catch] unwind to caller

catch:                                            ; preds = %dispatch
  %0 = catchpad within %cs1 [i8* null]
  catchret from %0 to label %return

fallthrough:                                      ; preds = %try
  unreachable

return:                                           ; preds = %catch, %entry
  ret void
}

; CHECK-LABEL: foo: # @foo
; CHECK: testb $1, %cl
; CHECK: jne .LBB0_[[return:[0-9]+]]
; CHECK: .Ltmp0:
; CHECK: callq bar
; CHECK: .Ltmp1:
; CHECK: .LBB0_[[catch:[0-9]+]]:
; CHECK: .LBB0_[[return]]:

; CHECK: .seh_handlerdata
; CHECK-NEXT: .long   (.Llsda_end0-.Llsda_begin0)/16
; CHECK-NEXT: .Llsda_begin0:
; CHECK-NEXT: .long   .Ltmp0@IMGREL+1
; CHECK-NEXT: .long   .Ltmp1@IMGREL+1
; CHECK-NEXT: .long   1
; CHECK-NEXT: .long   .LBB0_[[catch]]@IMGREL
