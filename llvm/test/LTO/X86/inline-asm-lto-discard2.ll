; Check that
; 1. ".lto_discard" works as module inlineasm marker and its argument symbols
;    are discarded.
; 2. there is no reassignment error in the presence of ".lto_discard"
; RUN: llc < %s | FileCheck %s

; CHECK:    .data
; CHECK-NOT:  .weak  foo
; CHECK-NOT:  .set   foo, bar
; CHECK:      .globl foo
; CHECK:      foo:
; CHECK:        .byte 1

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

module asm ".lto_discard foo"
module asm "	.text"
module asm "bar:"
module asm "	.data"
module asm ".weak foo"
module asm ".set   foo, bar"
module asm ".weak foo"
module asm ".set   foo, bar"

module asm ".lto_discard"
module asm ".globl foo"
module asm "foo:"
module asm "   .byte 1"
