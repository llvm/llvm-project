; RUN: llc < %s -mtriple=i686-pc-linux-gnu | FileCheck %s

@__gthrw_pthread_once = weak alias i32 (i32*, void ()*), i32 (i32*, void ()*)* @pthread_once		; <i32 (i32*, void ()*)*> [#uses=0]

define weak i32 @pthread_once(i32*, void ()*) {
  ret i32 0
}

; CHECK: .weak   pthread_once
; CHECK: pthread_once:

; CHECK: .weak   __gthrw_pthread_once
; CHECK: .set __gthrw_pthread_once, pthread_once
