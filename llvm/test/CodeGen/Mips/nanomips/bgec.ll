; RUN: llc -mtriple=nanomips -verify-machineinstrs -stop-after=finalize-isel < %s | FileCheck %s

define void @bgec_zero(i32 signext %arg) {
entry:
; CHECK: name:            bgec_zero
; CHECK: bb.0.entry:
; CHECK-NOT: %1:gpr32nm = Li_NM -1
; CHECK-NOT: BLTC_NM killed %1, %0, %bb.2
; CHECK: BGEC_NM %0, $zero_nm, %bb.2
; CHECK: BC_NM %bb.1
  %cmp = icmp slt i32 %arg, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:
  tail call void bitcast (void (...)* @bar to void ()*)()
  br label %if.end

if.else:
  tail call void bitcast (void (...)* @bat to void ()*)()
  br label %if.end

if.end:
  ret void
}

declare void @bar(...)
declare void @bat(...)
