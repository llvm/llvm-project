; RUN: llc -mtriple=arm-eabi -print-machineinstrs=tailduplication -tail-dup-size=100 \
; RUN:      -enable-tail-merge=false -disable-cgp %s -o /dev/null 2>&1 \
; RUN:	| FileCheck %s

; CHECK: Machine code for function test0:
; CHECK: Successors according to CFG: BB#1({{[0-9a-fx/= ]+}}3.12%) BB#2({{[0-9a-fx/= ]+}}96.88%)

define void @test0(i32 %a, i32 %b, i32* %c, i32* %d) {
entry:
  store i32 3, i32* %d
  br label %B1

B2:
  store i32 2, i32* %c
  br label %B4

B3:
  store i32 2, i32* %c
  br label %B4

B1:
  store i32 1, i32* %d
  %test0 = icmp slt i32 %a, %b
  br i1 %test0, label %B2, label %B3, !prof !0

B4:
  ret void
}

!0 = !{!"branch_weights", i32 4, i32 124}

; CHECK: Machine code for function test1:
; CHECK: Successors according to CFG: BB#1({{[0-9a-fx/= ]+}}3.12%) BB#2({{[0-9a-fx/= ]+}}96.88%)

@g0 = common global i32 0, align 4

define void @test1(i32 %a, i32 %b, i32* %c, i32* %d, i32* %e) {

  %test0 = icmp slt i32 %a, %b
  br i1 %test0, label %B1, label %B2, !prof !1

B1:
  br label %B3

B2:
  store i32 2, i32* %c
  br label %B3

B3:
  store i32 3, i32* %e
  ret void
}

!1 = !{!"branch_weights", i32 248, i32 8}
