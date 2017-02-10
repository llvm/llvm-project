; RUN: llc -mtriple=aarch64 -verify-machineinstrs < %s | FileCheck --check-prefix=CHECK --check-prefix=OPT %s
; RUN: llc -mtriple=aarch64 -verify-machineinstrs -O0 -fast-isel=0 < %s | FileCheck --check-prefix=CHECK --check-prefix=NOOPT %s

declare void @foo()

; Check that the inverted or doesn't inhibit the splitting of the
; complex conditional into three branch instructions.
; CHECK-LABEL: test_and_not:
; CHECK:       cbz w0, [[L:\.LBB[0-9_]+]]
; OPT:         cmp w1, #2
; NOOPT:       subs w{{[0-9]+}}, w{{[0-9]+}}, #2
; CHECK:       b.lo [[L]]
; OPT:         cmp w2, #2
; NOOPT:       subs w{{[0-9]+}}, w{{[0-9]+}}, #2
; CHECK:       b.hi [[L]]
define void @test_and_not(i32 %a, i32 %b, i32 %c) {
bb1:
  %cmp1 = icmp ult i32 %a, 1
  %cmp2 = icmp ult i32 %b, 2
  %cmp3 = icmp ult i32 %c, 3
  %or = or i1 %cmp1, %cmp2
  %not.or = xor i1 %or, -1
  %and = and i1 %not.or, %cmp3
  br i1 %and, label %bb2, label %bb3

bb2:
  ret void

bb3:
  call void @foo()
  ret void
}

; Check that non-canonicalized xor not is handled correctly by FindMergedConditions.
; CHECK-LABEL: test_and_not2:
; CHECK:       cbz w0, [[L:\.LBB[0-9_]+]]
; OPT:         cmp w1, #2
; NOOPT:       subs w{{[0-9]+}}, w{{[0-9]+}}, #2
; CHECK:       b.lo [[L]]
; OPT:         cmp w2, #2
; NOOPT:       subs w{{[0-9]+}}, w{{[0-9]+}}, #2
; CHECK:       b.hi [[L]]
define void @test_and_not2(i32 %a, i32 %b, i32 %c) {
bb1:
  %cmp1 = icmp ult i32 %a, 1
  %cmp2 = icmp ult i32 %b, 2
  %cmp3 = icmp ult i32 %c, 3
  %or = or i1 %cmp1, %cmp2
  %not.or = xor i1 -1, %or
  %and = and i1 %not.or, %cmp3
  br i1 %and, label %bb2, label %bb3

bb2:
  ret void

bb3:
  call void @foo()
  ret void
}

; Check that cmps in different blocks are handled correctly by FindMergedConditions.
; CHECK-LABEL: test_cmp_other_block:
; OPT: cmp w{{[0-9]+}}, #0
; OPT: b.gt [[L:\.LBB[0-9_]+]]
; OPT: tbz w1, #0, [[L]]
;
; NOOPT: subs w{{[0-9]+}}, w{{[0-9]+}}, #0
; NOOPT: cset [[R1:w[0-9]+]], gt
; NOOPT: str w1, [sp, #[[SLOT2:[0-9]+]]]
; NOOPT: str [[R1]], [sp, #[[SLOT1:[0-9]+]]]
; NOOPT: b .LBB
; NOOPT: ldr [[R2:w[0-9]+]], [sp, #[[SLOT1]]]
; NOOPT: tbnz [[R2]], #0, [[L:\.LBB[0-9_]+]]
; NOOPT: ldr [[R3:w[0-9]+]], [sp, #[[SLOT2]]]
; NOOPT: tbz [[R3]], #0, [[L]]
define void @test_cmp_other_block(i32* %p, i1 %c) {
entry:
  %l = load i32, i32* %p
  %cmp = icmp sgt i32 %l, 0
  br label %bb1

bb1:
  %cmp.i = xor i1 %cmp, true
  %or.cond1.i = and i1 %cmp.i, %c
  br i1 %or.cond1.i, label %bb2, label %bb3

bb2:
  ret void

bb3:
  call void @foo()
  ret void
}

