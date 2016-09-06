; RUN: llc -verify-machineinstrs < %s | grep "no_dead_strip.*_X"

target datalayout = "E-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "powerpc-apple-darwin8.8.0"
@X = weak global i32 0          ; <i32*> [#uses=1]
@.str = internal constant [4 x i8] c"t.c\00", section "llvm.metadata"          ; <[4 x i8]*> [#uses=1]
@llvm.used = appending global [1 x i8*] [ i8* bitcast (i32* @X to i8*) ], section "llvm.metadata"       ; <[1 x i8*]*> [#uses=0]

