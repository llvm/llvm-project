; RUN: llvm-as -o %t %s
; RUN: llvm-lto2 dump-symtab %t | FileCheck %s

; CHECK: target triple: i686-pc-windows-msvc18.0.0
target triple = "i686-pc-windows-msvc18.0.0"
target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"

; CHECK: source filename: src.c
source_filename = "src.c"

; CHECK: linker opts: /include:foo
!0 = !{!"/include:foo"}
!llvm.linker.options = !{ !0 }

; CHECK: D------X _fun
define i32 @fun() {
  ret i32 0
}

; CHECK: H------- _g1
@g1 = hidden global i32 0

; CHECK: P------- _g2
@g2 = protected global i32 0

; CHECK: D------- _g3
@g3 = global i32 0

; CHECK: DU------ _g4
@g4 = external global i32

; CHECK: D--W---- _g5
@g5 = weak global i32 0

; CHECK: D--W-O-- _g6
@g6 = linkonce_odr unnamed_addr global i32 0

; CHECK: D-----T- _g7
@g7 = thread_local global i32 0

; CHECK: D-C----- _g8
; CHECK-NEXT: size 4 align 8
@g8 = common global i32 0, align 8

; CHECK: D------- _g9
; CHECK-NEXT: comdat g9
$g9 = comdat any
@g9 = global i32 0, comdat

; CHECK: D--WI--- _g10
; CHECK-NEXT: comdat g9
; CHECK-NEXT: fallback _g9
@g10 = weak alias i32, i32* @g9
