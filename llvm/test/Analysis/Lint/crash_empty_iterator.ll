; RUN: opt -passes="lint" -S < %s | FileCheck %s

; After 2fe81edef6f0b
;  [NFC][RemoveDIs] Insert instruction using iterators in Transforms/
; this crashed in FindInsertedValue when dereferencing an empty
; optional iterator.
; Just see that it doesn't crash anymore.

; CHECK-LABEL: @test1

%struct = type { i32, i32 }

define void @test1() {
entry:
  %.fca.1.insert = insertvalue %struct zeroinitializer, i32 0, 1
  %0 = extractvalue %struct %.fca.1.insert, 0
  %1 = tail call %struct @foo(i32 %0)
  ret void
}

declare %struct @foo(i32)

