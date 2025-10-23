;; Verify the output of the hexinsert pass, llc will abort if verification
;; fails. In the failing case, a def (%15) is used in an bb that does
;; not follow the one with the def, e.g.:

;; # | bb.3 (%ir-block.3):
;; # |   %17:intregs = S2_insert %15:intregs(tied-def 0), %5:intregs, 5, 5
;; # |   J2_jump %bb.2, implicit-def dead $pc
;; # |
;; # | bb.4 (%ir-block.6):
;; # |   %15:intregs = S2_insert %6:intregs(tied-def 0), %3:intregs, 5, 5

; RUN: llc -O2 -mtriple=hexagon -insert-max-ifmap=1 -stop-after hexinsert -verify-machineinstrs < %s
; RUN: llc -O2 -mtriple=hexagon -insert-max-ifmap=2 -stop-after hexinsert -verify-machineinstrs < %s
; RUN: llc -O2 -mtriple=hexagon -insert-max-ifmap=3 -stop-after hexinsert -verify-machineinstrs < %s
; RUN: llc -O2 -mtriple=hexagon -insert-max-ifmap=4 -stop-after hexinsert -verify-machineinstrs < %s
; RUN: llc -O2 -mtriple=hexagon -insert-max-ifmap=5 -stop-after hexinsert -verify-machineinstrs < %s

define i32 @f(i32 %0, i32 %1, i32 %2) {
entry:
  switch i32 %0, label %common.ret1 [
    i32 8907, label %3
    i32 4115, label %6
  ]

common.ret1:
  %common.ret1.op = phi i32 [ %5, %3 ], [ %526, %6 ], [ 0, %entry ]
  ret i32 %common.ret1.op

3:
  %4 = shl i32 %2, 5
  %5 = and i32 %4, 992
  br label %common.ret1

6:
  %7 = shl i32 %0, 10
  %8 = and i32 %7, 7168
  %9 = shl i32 %0, 5
  %10 = and i32 %9, 992
  %11 = or i32 %10, %8
  %12 = and i32 %0, 1
  %13 = or i32 %11, %12
  %14 = shl i32 %1, 1
  %15 = and i32 %14, 2031616
  %526 = or i32 %13, %15
  br label %common.ret1
}
