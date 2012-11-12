; RUN: llc < %s -mtriple=i386-apple-darwin | not grep comm
; rdar://6479858

@str1 = internal constant [1 x i8] zeroinitializer
