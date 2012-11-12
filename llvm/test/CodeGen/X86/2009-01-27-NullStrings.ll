; RUN: llc < %s -mtriple=i686-apple-darwin | FileCheck %s
; CHECK: .section __TEXT,__cstring,cstring_literals

@x = internal unnamed_addr constant [1 x i8] zeroinitializer		; <[1 x i8]*> [#uses=1]

@y = global [1 x i8]* @x

