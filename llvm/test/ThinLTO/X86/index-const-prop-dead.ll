; RUN: opt -module-summary %s -o %t1.bc
; RUN: opt -module-summary %p/Inputs/index-const-prop-define-g.ll -o %t2.bc
; RUN: llvm-lto2 run -save-temps %t2.bc -r=%t2.bc,g,pl \
; RUN:               %t1.bc -r=%t1.bc,main,plx -r=%t1.bc,foo,pl -r=%t1.bc,g, -o %t3
; RUN: llvm-dis %t3.2.3.import.bc -o - | FileCheck %s

; Dead globals are converted to declarations by ThinLTO in dropDeadSymbols
; If we try to internalize such we'll get a broken module. 
; CHECK: @g = external global i32

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@g = external global i32

; We need at least one live symbol to enable dead stripping
; Otherwise ModuleSummaryIndex::isGlobalValueLive will always
; return true.
define i32 @main() {
  ret i32 42
}

define i32 @foo() {
  %v = load i32, ptr @g
  ret i32 %v
}
