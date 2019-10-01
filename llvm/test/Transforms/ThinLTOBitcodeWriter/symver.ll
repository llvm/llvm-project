; RUN: opt -thinlto-bc -thinlto-split-lto-unit -o %t %s
; RUN: llvm-modextract -n 1 -o - %t | llvm-dis | FileCheck %s

; The target assembly parser is required to parse the symver directives
; REQUIRES: x86-registered-target

target triple = "x86_64-unknown-linux-gnu"

module asm ".symver used, used@VER"
module asm ".symver unused, unused@VER"
module asm ".symver variable, variable@VER"

declare !type !0 void @used()
declare !type !0 void @unused()
@variable = global i32 0

define i32* @use() {
  call void @used()
  ret i32* @variable
}

; CHECK: !symvers = !{![[SYMVER:[0-9]+]]}
; CHECK: ![[SYMVER]] = !{!"used", !"used@VER"}

!0 = !{i64 0, !"_ZTSFvvE"}
