; RUN: llvm-link %s %p/unnamed-addr1-b.ll -S -o - | sort | FileCheck %s

; Only in this file
@a = common global i32 0
; CHECK: @a = common global i32 0
@b = common unnamed_addr global i32 0
; CHECK: @b = common unnamed_addr global i32 0

; Other file has unnamed_addr definition
@c = common unnamed_addr global i32 0
; CHECK: @c = common unnamed_addr global i32 0
@d = external global i32
; CHECK: @d = unnamed_addr global i32 42
@e = external unnamed_addr global i32
; CHECK: @e = unnamed_addr global i32 42
@f = weak global i32 42
; CHECK: @f = unnamed_addr global i32 42

; Other file has non-unnamed_addr definition
@g = common unnamed_addr global i32 0
; CHECK: @g = common unnamed_addr global i32 0
@h = external global i32
; CHECK: @h = global i32 42
@i = external unnamed_addr global i32
; CHECK: @i = global i32 42
@j = weak global i32 42
; CHECK: @j = global i32 42
