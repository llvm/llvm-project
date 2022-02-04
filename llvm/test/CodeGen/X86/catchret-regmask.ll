; RUN: llc < %s | FileCheck %s

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

declare i32 @__CxxFrameHandler3(...)
declare void @throw() noreturn uwtable
declare i8* @getval()
declare void @llvm.trap()

define i8* @reload_out_of_pad(i8* %arg) #0 personality i32 (...)* @__CxxFrameHandler3 {
assertPassed:
  invoke void @throw()
          to label %unreachable unwind label %catch.dispatch

catch:
  %cp = catchpad within %cs [i8* null, i32 0, i8* null]
  catchret from %cp to label %return

  ; This block *must* appear after the catchret to test the bug.
  ; FIXME: Make this an MIR test so we can control MBB layout.
unreachable:
  call void @llvm.trap()
  unreachable

catch.dispatch:
  %cs = catchswitch within none [label %catch] unwind to caller

return:
  ret i8* %arg
}

; CHECK-LABEL: reload_out_of_pad: # @reload_out_of_pad
; CHECK: movq %rcx, -[[arg_slot:[0-9]+]](%rbp) # 8-byte Spill
; CHECK: callq throw
; CHECK: ud2
; CHECK: movq -[[arg_slot]](%rbp), %rax # 8-byte Reload
; CHECK: retq

; CHECK: "?catch${{[0-9]+}}@?0?reload_out_of_pad@4HA":
; CHECK-NOT: Reload
; CHECK: retq

define i8* @spill_in_pad() #0 personality i32 (...)* @__CxxFrameHandler3 {
assertPassed:
  invoke void @throw()
          to label %unreachable unwind label %catch.dispatch

catch:
  %cp = catchpad within %cs [i8* null, i32 0, i8* null]
  %val = call i8* @getval() [ "funclet"(token %cp) ]
  catchret from %cp to label %return

unreachable:
  call void @llvm.trap()
  unreachable

catch.dispatch:
  %cs = catchswitch within none [label %catch] unwind to caller

return:
  ret i8* %val
}

; CHECK-LABEL: spill_in_pad: # @spill_in_pad
; CHECK: callq throw
; CHECK: ud2
; CHECK: movq -[[val_slot:[0-9]+]](%rbp), %rax # 8-byte Reload
; CHECK: retq

; CHECK: "?catch${{[0-9]+}}@?0?spill_in_pad@4HA":
; CHECK: callq getval
; CHECK: movq %rax, -[[val_slot]](%rbp) # 8-byte Spill
; CHECK: retq

attributes #0 = { uwtable }
