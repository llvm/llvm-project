; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-unknown-linux-gnu | FileCheck %s

@X = global <{i32, i32}> <{ i32 1, i32 123 }>

; CHECK: align 3
