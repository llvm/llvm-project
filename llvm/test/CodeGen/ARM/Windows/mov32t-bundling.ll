; RUN: llc -mtriple thumbv7-windows-itanium -filetype asm -o - %s | FileCheck %s

@_begin = external global i8
@_end = external global i8

declare arm_aapcs_vfpcc void @force_emission()

define arm_aapcs_vfpcc void @bundle() {
entry:
  %cmp = icmp uge i32 sub (i32 ptrtoint (ptr @_end to i32), i32 ptrtoint (ptr @_begin to i32)), 4
  br i1 %cmp, label %if.then, label %if.end

if.then:
  tail call arm_aapcs_vfpcc void @force_emission()
  br label %if.end

if.end:
  ret void
}

; CHECK-LABEL: bundle
; CHECK-NOT: subs r0, r1, r0
; CHECK: movw r0, :lower16:_begin
; CHECK-NEXT: movt r0, :upper16:_begin
; CHECK-NEXT: movw r1, :lower16:_end
; CHECK-NEXT: movt r1, :upper16:_end
; CHECK-NEXT: subs r0, r1, r0
; CHECK-NEXT: cmp r0, #4

