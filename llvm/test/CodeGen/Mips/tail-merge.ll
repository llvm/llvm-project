; RUN: split-file %s %t
; RUN: llc -mtriple=mipsel -verify-machineinstrs < %t/test.ll | FileCheck %s
; RUN: llc -mtriple=mipsel -relocation-model=pic -force-mips-long-branch -verify-machineinstrs < %t/test.ll | FileCheck %s
; RUN: llc -mtriple=mipsel -mattr=+reserve-gpr1 -verify-machineinstrs < %t/test.ll | FileCheck %s --check-prefix=SAFE
; RUN: llc -mtriple=mipsel -run-pass=branch-folder -enable-tail-merge -verify-machineinstrs %t/reuse.mir -o - | FileCheck %s --check-prefix=REUSE
; RUN: llc -mtriple=mipsel -mattr=+mips16,+soft-float -verify-machineinstrs < %t/test.ll | FileCheck %s --check-prefix=MIPS16

; Keep the two tails separate when merging would leave $at live across a branch.
; Reserving $at makes the same tail safe to merge.
; SAFE: addiu ${{[0-9]+}}, ${{[0-9]+}}, 23
; SAFE-NOT: addiu ${{[0-9]+}}, ${{[0-9]+}}, 23

; MIPS16 keeps tail merging disabled.
; MIPS16: addiu ${{[0-9]+}}, 23
; MIPS16: addiu ${{[0-9]+}}, 23

;--- test.ll

@g0 = common global i32 0, align 4
@g1 = common global i32 0, align 4

; CHECK: addiu ${{[0-9]+}}, ${{[0-9]+}}, 23
; CHECK: addiu ${{[0-9]+}}, ${{[0-9]+}}, 23

define i32 @test1(i32 %a) {
entry:
  %tobool = icmp eq i32 %a, 0
  %0 = load i32, ptr @g0, align 4
  br i1 %tobool, label %if.else, label %if.then

if.then:
  %add = add nsw i32 %0, 1
  store i32 %add, ptr @g0, align 4
  %1 = load i32, ptr @g1, align 4
  %add1 = add nsw i32 %1, 23
  br label %if.end

if.else:
  %add2 = add nsw i32 %0, 11
  store i32 %add2, ptr @g0, align 4
  %2 = load i32, ptr @g1, align 4
  %add3 = add nsw i32 %2, 23
  br label %if.end

if.end:
  %storemerge = phi i32 [ %add3, %if.else ], [ %add1, %if.then ]
  store i32 %storemerge, ptr @g1, align 4
  ret i32 %storemerge
}

;--- reuse.mir
# A whole-block common tail must obey the same legality check as a split tail.
# The first shared instruction does not use $at, but a later instruction does.
# REUSE-LABEL: name: reuse
# REUSE: $at = ADDiu $a1, {{[12]}}
# REUSE-NEXT: $v0 = LW $a3, 0
# REUSE-NEXT: $v0 = ADDu $v0, $at
# REUSE: $at = ADDiu $a1, {{[12]}}
# REUSE-NEXT: $v0 = LW $a3, 0
# REUSE-NEXT: $v0 = ADDu $v0, $at
---
name: reuse
tracksRegLiveness: true
body: |
  bb.0:
    successors: %bb.1, %bb.2
    liveins: $a0, $a1, $a3
    BEQ $a0, $zero, %bb.2, implicit-def $at
    B %bb.1, implicit-def $at

  bb.1:
    liveins: $a1, $a3
    $at = ADDiu $a1, 1
    $v0 = LW $a3, 0
    $v0 = ADDu $v0, $at
    RetRA implicit $v0

  bb.2:
    successors: %bb.3
    liveins: $a1, $a3
    $at = ADDiu $a1, 2

  bb.3:
    liveins: $at, $a3
    $v0 = LW $a3, 0
    $v0 = ADDu $v0, $at
    RetRA implicit $v0
...
