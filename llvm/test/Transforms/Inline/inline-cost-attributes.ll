; REQUIRES: asserts
; RUN: opt -inline-cost-full -passes='cgscc(inline)' -debug-only=inline -disable-output %s 2>&1 | FileCheck --check-prefix=INLINER %s
; RUN: opt -inline-cost-full -passes='print<inline-cost>' -disable-output %s 2>&1 | FileCheck --check-prefix=COST %s

declare void @extern() "call-threshold-bonus"="31"

define void @fn1() "function-inline-cost"="321" "function-inline-threshold"="123" "call-inline-cost"="271" {
entry:
  ret void
}

define void @fn2() "function-inline-threshold"="41" {
; INLINER-LABEL: Inlining calls in: fn2
; INLINER-NEXT: Function size: 6
; INLINER-NEXT: NOT Inlining (cost=321, threshold=123), Call:   call void @fn1()
; INLINER-NEXT: NOT Inlining (cost=321, threshold=321), Call:   call void @fn1()
; INLINER-NEXT: NOT Inlining (cost=197, threshold=123), Call:   call void @fn1()
; INLINER-NEXT: Inlining (cost=197, threshold=321), Call:   call void @fn1()

; COST-LABEL: define void @fn2()
; COST-NEXT: entry:
; COST-NEXT: threshold delta = 31
; COST-NEXT: call void @extern()
; COST-NEXT: cost delta = 132, threshold delta = 193
; COST-NEXT: call void @fn1()
; COST-NEXT: cost delta = 0
; COST-NEXT: call void @fn1()
; COST-NEXT: cost delta = 271, threshold delta = 17
; COST-NEXT: call void @fn1()
; COST-NEXT: cost delta = 473
; COST-NEXT: call void @fn1()

entry:
  call void @extern()
  call void @fn1() "call-inline-cost"="132" "call-threshold-bonus"="193"
  call void @fn1() "call-inline-cost"="0" "function-inline-threshold"="321"
  call void @fn1() "call-threshold-bonus"="17" "function-inline-cost"="197"
  call void @fn1() "call-inline-cost"="473" "function-inline-cost"="197" "function-inline-threshold"="321"
  ret void
}

define void @fn3() {
; INLINER-LABEL: Inlining calls in: fn3
; INLINER-NEXT: Function size: 3
; INLINER-NEXT: Inlining (cost=386, threshold=849), Call:   call void @fn1()
; INLINER-NEXT: Size after inlining: 2
; INLINER-NEXT: NOT Inlining (cost=403, threshold=41), Call:   call void @fn2()

entry:
  call void @fn1() "function-inline-cost"="386" "function-inline-threshold"="849"
  call void @fn2()
  ret void
}
