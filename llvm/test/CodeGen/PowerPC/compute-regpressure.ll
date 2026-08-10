; REQUIRES: asserts
; RUN: llc -debug-only=regalloc < %s 2>&1 |FileCheck %s --check-prefix=DEBUG

; DEBUG-COUNT-1:         AllocationOrder(VRSAVERC) = [ ]

target triple = "powerpc64le-unknown-linux-gnu"

define hidden fastcc void @test() {
freescalar:
  %0 = load i32, ptr undef, align 4
  br label %if.end420

if.end420:                                        ; preds = %freescalar
  br label %free_rv

free_rv:                                          ; preds = %if.end420
  %and427 = and i32 %0, -2147481600
  %cmp428 = icmp eq i32 %and427, -2147481600
  br i1 %cmp428, label %if.then430, label %free_body

if.then430:                                       ; preds = %free_rv
  call fastcc void undef()
  br label %free_body

free_body:                                        ; preds = %if.then430, %free_rv
  %or502 = or i32 undef, 255
  store i32 %or502, ptr undef, align 4
  ret void
}

