; RUN: opt -S -bdce < %s | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

define void @PR34211(i16* %p) {
; CHECK-LABEL: @PR34211
  %not_demanded_but_not_dead = load volatile i16, i16* %p
  call void @no_side_effects_so_dead(i16 %not_demanded_but_not_dead)
  ret void

; CHECK: %not_demanded_but_not_dead = load volatile i16, i16* %p
; CHECK-NEXT: ret void
}

declare void @no_side_effects_so_dead(i16) #0

attributes #0 = { nounwind readnone willreturn }

