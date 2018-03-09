; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-apple-darwin8.8.0 | \
; RUN:   grep align.*3

@X = global <{i32, i32}> <{ i32 1, i32 123 }>
