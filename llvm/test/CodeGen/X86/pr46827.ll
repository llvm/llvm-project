; RUN: llc < %s -mtriple=i686-pc-linux -mattr=+rtm -verify-machineinstrs -stop-after=finalize-isel | FileCheck %s

; CHECK: body:             |
; CHECK:   bb.0.bb107:
; CHECK:     successors: %bb.3(0x40000000), %bb.4(0x40000000)
; CHECK:     %0:gr32 = MOV32rm %fixed-stack.0, 1, $noreg, 0, $noreg :: (load (s32) from %fixed-stack.0, align 16)
; CHECK:     %1:gr32 = SUB32ri %0, 1, implicit-def $eflags
; CHECK:     XBEGIN_4 %bb.4, implicit-def $eax
; CHECK:   bb.3.bb107:
; CHECK:     successors: %bb.5(0x80000000)
; CHECK:     liveins: $eflags
; CHECK:     %3:gr32 = MOV32ri -1
; CHECK:     JMP_1 %bb.5
; CHECK:   bb.4.bb107:
; CHECK:     successors: %bb.5(0x80000000)
; CHECK:     liveins: $eflags
; CHECK:     XABORT_DEF implicit-def $eax
; CHECK:     %4:gr32 = COPY $eax
; CHECK:   bb.5.bb107:
; CHECK:     successors: %bb.1(0x40000000), %bb.2(0x40000000)
; CHECK:     liveins: $eflags
; CHECK:     %2:gr32 = PHI %3, %bb.3, %4, %bb.4
; CHECK:     JCC_1 %bb.2, 5, implicit $eflags
; CHECK:     JMP_1 %bb.1

declare i32 @llvm.x86.xbegin() #0

define void @wobble.12(i32 %tmp116) {
bb107:                                            ; preds = %bb42
  %tmp117 = icmp eq i32 %tmp116, 1
  %tmp127 = tail call i32 @llvm.x86.xbegin() #0
  br i1 %tmp117, label %bb129, label %bb250

bb129:                                            ; preds = %bb107
  unreachable

bb250:                                            ; preds = %bb107
  unreachable
}
