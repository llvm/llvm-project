; RUN: llvm-as %s -o %t1.bc
; RUN: llvm-as %p/Inputs/global-inline-asm-flags.ll -o %t2.bc

; RUN: llvm-lto -save-merged-module -filetype=asm -mattr=+pauth %t1.bc %t2.bc -o %t3
; RUN: llvm-dis %t3.merged.bc -o - | FileCheck %s

; RUN: llvm-lto2 run -save-temps -mattr=+pauth -filetype=asm -o %t4.s %t1.bc %t2.bc \
; RUN:   -r=%t1.bc,baz,p \
; RUN:   -r=%t1.bc,baz@VER,p \
; RUN:   -r=%t1.bc,foo@LINKEDVER,p \
; RUN:   -r=%t2.bc,bar,p \
; RUN:   -r=%t2.bc,bar@VER,p \
; RUN:   -r=%t2.bc,foo@ANOTHERVER,p \
; RUN:   -r=%t2.bc,foo,p \
; RUN:   -r=%t2.bc,foo@VER,p
; RUN: llvm-dis %t4.s.0.5.precodegen.bc -o - | FileCheck %s

; Note that -mattr=+pauth option for llvm-lto and llvm-lto2 is still
; required, because LTO runs full CodeGen at the end. Symbols and
; Symvers are still extracted from metadata.

; RUN: llvm-nm %t1.bc | FileCheck %s --check-prefix NM1
; RUN: llvm-nm %t2.bc | FileCheck %s --check-prefix NM2
; RUN: llvm-nm %t3.merged.bc | FileCheck %s --check-prefixes NM1,NM2
; RUN: llvm-nm %t4.s.0.5.precodegen.bc | FileCheck %s --check-prefixes NM1,NM2

; Symbols of the first module
; NM1-DAG: T baz
; NM1-DAG: T baz@VER
; NM1-DAG: T foo@LINKEDVER

; Symbols of the second module
; NM2-DAG: T bar
; NM2-DAG: T bar@VER
; NM2-DAG: T foo
; NM2-DAG: T foo@ANOTHERVER
; NM2-DAG: T foo@VER

; IR with two modules linked
; CHECK: module asm ".text"
; CHECK: module asm ".balign 16"
; CHECK: module asm ".globl baz"
; CHECK: module asm "baz:"
; CHECK: module asm "pacib     x30, x27"
; CHECK: module asm "retab"
; CHECK: module asm ".symver baz, baz@VER"
; CHECK: module asm ".symver foo, foo@LINKEDVER"
; CHECK: module asm ".previous"
; CHECK: module asm ".text"
; CHECK: module asm ".balign 16"
; CHECK: module asm ".globl foo"
; CHECK: module asm "foo:"
; CHECK: module asm "pacib     x30, x27"
; CHECK: module asm "retab"
; CHECK: module asm ".symver foo, foo@VER"
; CHECK: module asm ".symver foo, foo@ANOTHERVER"
; CHECK: module asm ".globl bar"
; CHECK: module asm "bar:"
; CHECK: module asm "pacib     x30, x27"
; CHECK: module asm "retab"
; CHECK: module asm ".symver bar, bar@VER"
; CHECK: module asm ".previous"

; CHECK: !{{[0-9]+}} = distinct !{i32 6, !"global-asm-symbols", ![[SYM:[0-9]+]]}
; CHECK: ![[SYM]] = distinct !{![[SBAZ1:[0-9]+]], ![[SBAZ2:[0-9]+]], ![[SFOO1:[0-9]+]], ![[SBAR1:[0-9]+]], ![[SBAR2:[0-9]+]], ![[SFOO2:[0-9]+]], ![[SFOO3:[0-9]+]], ![[SFOO4:[0-9]+]]}
; CHECK: ![[SBAZ1]] = !{!"baz", i32 2050}
; CHECK: ![[SBAZ2]] = !{!"baz@VER", i32 2050}
; CHECK: ![[SFOO1]] = !{!"foo@LINKEDVER", i32 2050}
; CHECK: ![[SBAR1]] = !{!"bar", i32 2050}
; CHECK: ![[SBAR2]] = !{!"bar@VER", i32 2050}
; CHECK: ![[SFOO2]] = !{!"foo@ANOTHERVER", i32 2050}
; CHECK: ![[SFOO3]] = !{!"foo", i32 2050}
; CHECK: ![[SFOO4]] = !{!"foo@VER", i32 2050}

; CHECK: !{{[0-9]+}} = distinct !{i32 6, !"global-asm-symvers", ![[SYMVER:[0-9]+]]}
; CHECK: ![[SYMVER]] = distinct !{![[VBAZ:[0-9]+]], ![[VFOO1:[0-9]+]], ![[VFOO2:[0-9]+]], ![[VBAR:[0-9]+]]}
; CHECK: ![[VBAZ]] = !{!"baz", !"baz@VER"}
; CHECK: ![[VFOO1]] = !{!"foo", !"foo@LINKEDVER"}
; CHECK: ![[VFOO2]] = !{!"foo", !"foo@VER", !"foo@ANOTHERVER"}
; CHECK: ![[VBAR]] = !{!"bar", !"bar@VER"}

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

module asm ".text"
module asm ".balign 16"
module asm ".globl baz"
module asm "baz:"
module asm "pacib     x30, x27"
module asm "retab"
module asm ".symver baz, baz@VER"
module asm ".symver foo, foo@LINKEDVER"
module asm ".previous"

!llvm.module.flags = !{!0, !5}

!0 = !{i32 6, !"global-asm-symbols", !1}
!1 = !{!2, !3, !4}
!2 = !{!"baz", i32 2050}
!3 = !{!"baz@VER", i32 2050}
!4 = !{!"foo@LINKEDVER", i32 2050}
!5 = !{i32 6, !"global-asm-symvers", !6}
!6 = !{!7, !8}
!7 = !{!"baz", !"baz@VER"}
!8 = !{!"foo", !"foo@LINKEDVER"}
