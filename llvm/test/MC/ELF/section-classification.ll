; We run this test explicitly converting from `.ll` to `.s` to `.o` to test the
; classification of the section and the incorrect handling for the padding flag.

; RUN: llc -filetype asm -o - %s | llvm-mc -triple x86_64-unknown-linux-gnu -filetype obj -o /dev/null -
; REQUIRES: asserts

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.C = type { i32 }

$_ZZ6getRefvE8instance = comdat any

@_ZZ6getRefvE8instance = linkonce_odr dso_local global %struct.C zeroinitializer, comdat, align 4
@llvm.compiler.used = appending global [1 x ptr] [ptr @_ZZ6getRefvE8instance], section "llvm.metadata"
