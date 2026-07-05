; RUN: llc < %s -mtriple=x86_64-darwin -mcpu=generic | FileCheck %s
; XFAIL: *
; rdar:15661073 simple example of redundant adds
;
; MachineCSE should coalesce trivial subregister copies.
;
; The extra movl+addl should be removed during MachineCSE.
; CHECK-LABEL: redundantadd
; CHECK: cmpq
; CHECK: movq
; CHECK-NOT: movl
; CHECK: addl
; CHECK-NOT: addl
; CHECK: ret

define i64 @redundantadd(ptr %a0, ptr %a1) {
entry:
  %tmp8 = load i64, ptr %a0, align 8
  %tmp12 = load i64, ptr %a1, align 8
  %tmp13 = icmp ult i64 %tmp12, -281474976710656
  br i1 %tmp13, label %exit1, label %body

exit1:
  unreachable

body:
  %tmp14 = trunc i64 %tmp8 to i32
  %tmp15 = trunc i64 %tmp12 to i32
  %tmp16 = tail call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %tmp14, i32 %tmp15)
  %tmp17 = extractvalue { i32, i1 } %tmp16, 1
  br i1 %tmp17, label %exit2, label %return

exit2:
  unreachable

return:
  %tmp18 = add i64 %tmp12, %tmp8
  %tmp19 = and i64 %tmp18, 4294967295
  %tmp20 = or i64 %tmp19, -281474976710656
  ret i64 %tmp20
}

declare { i32, i1 } @llvm.sadd.with.overflow.i32(i32, i32)
