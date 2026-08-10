; RUN: opt -S < %s | FileCheck %s --check-prefixes=ALL,ALL_BUT_TWO,ALL_BUT_THREE,ALL_BUT_FOUR,ONE_AND_TWO,ONE_AND_THREE,ONE_AND_FOUR,ONE
; RUN: opt -S -passes=globalopt < %s | FileCheck %s --check-prefixes=ALL,ALL_BUT_ONE,ALL_BUT_THREE,ALL_BUT_FOUR,ONE_AND_TWO,TWO_AND_THREE,TWO_AND_FOUR,TWO
; RUN: opt -S -passes=instsimplify < %s | FileCheck %s --check-prefixes=ALL,ALL_BUT_ONE,ALL_BUT_TWO,ALL_BUT_FOUR,ONE_AND_THREE,TWO_AND_THREE,THREE_AND_FOUR,THREE
; RUN: opt -S < %s | FileCheck %s --check-prefixes=ALL,ALL_BUT_ONE,ALL_BUT_TWO,ALL_BUT_THREE,ONE_AND_FOUR,TWO_AND_FOUR,THREE_AND_FOUR,FOUR
;
; Make sure we don't use anything to check for @sometimes_here that contains "ALL" or "TWO".
; Also verify we use "ONE_AND_FOUR" for the unmodified @sometimes_here version and "THREE" for the version without the add.

define internal void @sometimes_here() {
  %c = add i32 undef, undef
  ret void
}

define void @always_here() {
  ret void
}
