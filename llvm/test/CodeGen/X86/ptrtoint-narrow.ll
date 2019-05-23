; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu | FileCheck %s

@ptr = external global i8, align 1
@ref = constant i32 ptrtoint (i8* @ptr to i32), align 4

; CHECK: .long  ptr{{$}}
