; RUN: llc -mtriple=xtensa -verify-machineinstrs < %s \
; RUN:   | FileCheck %s

; Check that we place const array (CTTZ lookup table) in global variable,
; instead of constant pool and place label to this table in constant pool.

; CHECK:      .literal_position
; CHECK-NEXT: .literal .LCPI0_0, 125613361
; CHECK-NEXT: .literal .LCPI0_1, __mulsi3
; CHECK-NEXT: .literal .LCPI0_2, .LCP0_0
; CHECK-NEXT: .global test_cttz_i32

define i32 @test_cttz_i32(i32 %a) nounwind {
; XTENSA-LABEL: test_cttz_i32:
; XTENSA:         addi a8, a1, -16
; XTENSA-NEXT:    or a1, a8, a8
; XTENSA-NEXT:    s32i a0, a1, 0 # 4-byte Folded Spill
; XTENSA-NEXT:    or a8, a2, a2
; XTENSA-NEXT:    movi a2, 32
; XTENSA-NEXT:    beqz a8, .LBB0_2
; XTENSA-NEXT:    j .LBB2_1
; XTENSA-NEXT:  .LBB2_1: # %cond.false
; XTENSA-NEXT:    neg a9, a8
; XTENSA-NEXT:    and a2, a8, a9
; XTENSA-NEXT:    l32r a3, .LCPI0_0
; XTENSA-NEXT:    l32r a8, .LCPI0_1
; XTENSA-NEXT:    callx0 a8
; XTENSA-NEXT:    movi a8, 27
; XTENSA-NEXT:    ssr a8
; XTENSA-NEXT:    srl a8, a2
; XTENSA-NEXT:    l32r a9, .LCPI0_2
; XTENSA-NEXT:    add a8, a9, a8
; XTENSA-NEXT:    l8ui a2, a8, 0
; XTENSA-NEXT:  .LBB2_2: # %cond.end
; XTENSA-NEXT:    l32i a0, a1, 0 # 4-byte Folded Reload
; XTENSA-NEXT:    addi a8, a1, 16
; XTENSA-NEXT:    or a1, a8, a8
; XTENSA-NEXT:    ret
  %tmp = call i32 @llvm.cttz.i32(i32 %a, i1 false)
  ret i32 %tmp
}

; CHECK:      .LCP0_0:
; CHECK-NEXT: .ascii "\000\001\034\002\035\016\030\003\036\026\024\017\031\021\004\b\037\033\r\027\025\023\020\007\032\f\022\006\013\005\n\t"
; CHECK-NEXT: .size  .LCP0_0, 32
