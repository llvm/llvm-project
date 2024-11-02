; RUN: llc < %s -mtriple=i686-pc-linux-gnu | FileCheck %s

@__gthrw_pthread_once = weak alias i32 (ptr, ptr), ptr @pthread_once		; <ptr> [#uses=0]

define weak i32 @pthread_once(ptr, ptr) {
  ret i32 0
}

; CHECK: .weak   pthread_once
; CHECK: pthread_once:

; CHECK: .weak   __gthrw_pthread_once
; CHECK: .set __gthrw_pthread_once, pthread_once
