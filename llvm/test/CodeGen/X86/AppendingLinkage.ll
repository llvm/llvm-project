; RUN: not llc < %s -mtriple=i686-- 2>&1 | FileCheck %s

; CHECK: unknown special variable
@foo = appending constant [1 x i32 ]zeroinitializer
