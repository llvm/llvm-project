; RUN: llc -mtriple=amdgpu9.00 -stop-after=si-wqm -o - %s | FileCheck %s

; A strict WWM entry that starts the block split off after a demote is a block
; prolog instruction.

; CHECK-LABEL: name: demote_then_wwm
; CHECK: {{^  bb\.1}}
; CHECK-NEXT: bb-prolog ENTER_STRICT_WWM

declare void @llvm.amdgcn.wqm.demote(i1)
declare i32 @llvm.amdgcn.strict.wwm.i32(i32)

define amdgpu_ps i32 @demote_then_wwm(i32 %x, i32 %c) {
  %cond = icmp ne i32 %c, 0
  call void @llvm.amdgcn.wqm.demote(i1 %cond)
  %v = add i32 %x, 1
  %r = call i32 @llvm.amdgcn.strict.wwm.i32(i32 %v)
  ret i32 %r
}
