; RUN: llc < %s -mtriple=x86_64-pc-windows-msvc -filetype=obj -o - | llvm-readobj -S - | FileCheck %s

@covmap = private global i32 0, section ".lcovmap$M"
@covfun = private global i32 0, section ".lcovfun$M"
@covname = private global i32 0, section ".lcovd"
@covdata= private global i32 0, section ".lcovn"

; CHECK: Name: .lcovmap$M
; CHECK: IMAGE_SCN_MEM_DISCARDABLE (0x2000000)
; CHECK: Name: .lcovfun$M
; CHECK: IMAGE_SCN_MEM_DISCARDABLE (0x2000000)
; CHECK: Name: .lcovd
; CHECK: IMAGE_SCN_MEM_DISCARDABLE (0x2000000)
; CHECK: Name: .lcovn
; CHECK: IMAGE_SCN_MEM_DISCARDABLE (0x2000000)
