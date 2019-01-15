; RUN: llc < %s -mtriple=thumb-apple-darwin -relocation-model=pic -frame-pointer=all | FileCheck %s -check-prefix=THUMB
; RUN: llc < %s -mtriple=arm-apple-darwin -relocation-model=pic -frame-pointer=all   | FileCheck %s -check-prefix=ARM
; RUN: llc < %s -mtriple=arm-apple-darwin -relocation-model=pic -frame-pointer=all -mattr=+v6t2 | FileCheck %s -check-prefix=MOVT
; rdar://7353541
; rdar://7354376
; rdar://8887598

@GV = external global i32                         ; <i32*> [#uses=2]

define void @t(i32* nocapture %vals, i32 %c) nounwind {
entry:
; ARM-LABEL: t:
; ARM: ldr [[REGISTER_1:r[0-9]+]], LCPI0_0
; ARM: LPC0_0:
; ARM: ldr r{{[0-9]+}}, [pc, [[REGISTER_1]]]
; ARM: ldr r{{[0-9]+}}, [r{{[0-9]+}}]

; MOVT-LABEL: t:
; MOVT: movw [[REGISTER_2:r[0-9]+]], :lower16:(L_GV$non_lazy_ptr-(LPC0_0+8))
; MOVT: movt [[REGISTER_2]], :upper16:(L_GV$non_lazy_ptr-(LPC0_0+8))
; MOVT: LPC0_0:
; MOVT: ldr r{{[0-9]+}}, [pc, [[REGISTER_2]]]
; MOVT: ldr r{{[0-9]+}}, [r{{[0-9]+}}]

; THUMB-LABEL: t:
  %0 = icmp eq i32 %c, 0                          ; <i1> [#uses=1]
  br i1 %0, label %return, label %bb.nph

bb.nph:                                           ; preds = %entry
; ARM: LCPI0_0:
; ARM-NOT: LCPI0_1:
; ARM: .section

; THUMB: %bb.1
; THUMB: ldr r2, LCPI0_0
; THUMB: add r2, pc
; THUMB: ldr r{{[0-9]+}}, [r2]
; THUMB: LBB0_2
; THUMB: LCPI0_0:
; THUMB-NOT: LCPI0_1:
; THUMB: .section
  %.pre = load i32, i32* @GV, align 4                  ; <i32> [#uses=1]
  br label %bb

bb:                                               ; preds = %bb, %bb.nph
  %1 = phi i32 [ %.pre, %bb.nph ], [ %3, %bb ]    ; <i32> [#uses=1]
  %i.03 = phi i32 [ 0, %bb.nph ], [ %4, %bb ]     ; <i32> [#uses=2]
  %scevgep = getelementptr i32, i32* %vals, i32 %i.03  ; <i32*> [#uses=1]
  %2 = load i32, i32* %scevgep, align 4                ; <i32> [#uses=1]
  %3 = add nsw i32 %1, %2                         ; <i32> [#uses=2]
  store i32 %3, i32* @GV, align 4
  %4 = add i32 %i.03, 1                           ; <i32> [#uses=2]
  %exitcond = icmp eq i32 %4, %c                  ; <i1> [#uses=1]
  br i1 %exitcond, label %return, label %bb

return:                                           ; preds = %bb, %entry
  ret void
}
