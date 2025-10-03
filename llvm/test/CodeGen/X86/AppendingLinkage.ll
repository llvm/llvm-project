; RUN: not llc < %s -mtriple=i686-- 2>&1 | FileCheck %s

; CHECK: error: unknown special variable with appending linkage: foo
@foo = appending constant [1 x i32 ] zeroinitializer

; CHECK: error: unknown special variable with appending linkage: @0
@0 = appending constant [1 x i32 ] zeroinitializer
