;; The bitcode symbol table parses module asm with the library options of the
;; module's context.
; RUN: not opt %s -o /dev/null 2>&1 | FileCheck %s
; RUN: opt -arm-implicit-it=always %s -o /dev/null

; CHECK: error: <inline asm>:3:1: predicated instructions must be in IT block

target triple = "thumbv7-unknown-linux-gnueabihf"

module asm ".syntax unified"
module asm ".thumb"
module asm "addeq r0, r0, r1"
