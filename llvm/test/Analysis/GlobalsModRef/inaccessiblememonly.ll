; RUN: opt -O3 -S < %s | FileCheck %s

target datalayout = "e-i64:64-f80:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

define void @donteliminate() {
; CHECK-LABEL: donteliminate
; CHECK-NEXT: tail call noalias ptr @allocmemory()
; CHECK-NEXT: tail call noalias ptr @allocmemory()
; CHECK-NEXT: tail call noalias ptr @allocmemory()
; CHECK-NEXT: ret void
  %1 = tail call noalias ptr @allocmemory()
  %2 = tail call noalias ptr @allocmemory()
  %3 = tail call noalias ptr @allocmemory()
  ret void
}

; Function Attrs: inaccessiblememonly
declare noalias ptr @allocmemory() #0

attributes #0 = { inaccessiblememonly }
