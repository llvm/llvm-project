; RUN: llc < %s -mtriple=i386-apple-darwin9 | grep no_dead_strip | count 1
; We should have a .no_dead_strip directive for Z but not for X/Y.

@X = internal global i8 4
@Y = internal global i32 123
@Z = internal global i8 4

@llvm.used = appending global [1 x i8*] [ i8* @Z ], section "llvm.metadata"
@llvm.compiler_used = appending global [2 x i8*] [ i8* @X, i8* bitcast (i32* @Y to i8*)], section "llvm.metadata"
