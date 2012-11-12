; This file is for use with unnamed-addr1-a.ll
; RUN: true

@c = common unnamed_addr global i32 42
@d = unnamed_addr global i32 42
@e = unnamed_addr global i32 42
@f = unnamed_addr global i32 42

@g = common global i32 42
@h = global i32 42
@i = global i32 42
@j = global i32 42
