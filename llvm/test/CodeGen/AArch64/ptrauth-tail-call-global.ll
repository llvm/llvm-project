; RUN: llc --mattr=+pauth -filetype=asm < %s | FileCheck %s

; CHECK:         adrp    x0, global
; CHECK-NEXT:    add x0, x0, :lo12:global
; CHECK-NEXT:    braaz   x0


target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-pauthtest"

@global = dso_local local_unnamed_addr global [256 x i8] zeroinitializer, align 16

define dso_local void @_Z3foov() local_unnamed_addr {
entry:
  tail call void @global() [ "ptrauth"(i32 0, i64 0) ]
  ret void
}

