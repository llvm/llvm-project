; RUN: llc < %s -march=x86    | FileCheck %s -check-prefix=X32
; RUN: llc < %s -march=x86-64 | FileCheck %s -check-prefix=X64
; rdar://7367229

define i32 @t(i32 %a, i32 %b) nounwind ssp {
entry:
; X32:     t:
; X32:     xorb
; X32-NOT: andb
; X32-NOT: shrb
; X32:     testb $64
; X32:     je

; X64:     t:
; X64-NOT: setne
; X64:     xorl
; X64:     testb $64
; X64:     je
  %0 = and i32 %a, 16384
  %1 = icmp ne i32 %0, 0
  %2 = and i32 %b, 16384
  %3 = icmp ne i32 %2, 0
  %4 = xor i1 %1, %3
  br i1 %4, label %bb1, label %bb

bb:                                               ; preds = %entry
  %5 = tail call i32 (...)* @foo() nounwind       ; <i32> [#uses=1]
  ret i32 %5

bb1:                                              ; preds = %entry
  %6 = tail call i32 (...)* @bar() nounwind       ; <i32> [#uses=1]
  ret i32 %6
}

declare i32 @foo(...)

declare i32 @bar(...)

define i32 @t2(i32 %x, i32 %y) nounwind ssp {
; X32: t2:
; X32: cmpl
; X32: sete
; X32: cmpl
; X32: sete
; X32-NOT: xor
; X32: je

; X64: t2:
; X64: testl
; X64: sete
; X64: testl
; X64: sete
; X64-NOT: xor
; X64: je
entry:
  %0 = icmp eq i32 %x, 0                          ; <i1> [#uses=1]
  %1 = icmp eq i32 %y, 0                          ; <i1> [#uses=1]
  %2 = xor i1 %1, %0                              ; <i1> [#uses=1]
  br i1 %2, label %bb, label %return

bb:                                               ; preds = %entry
  %3 = tail call i32 (...)* @foo() nounwind       ; <i32> [#uses=0]
  ret i32 undef

return:                                           ; preds = %entry
  ret i32 undef
}
