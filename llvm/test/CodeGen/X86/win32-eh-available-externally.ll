; RUN: opt -S -x86-winehstate < %s | FileCheck %s --check-prefix=IR
; RUN: llc < %s | FileCheck %s --check-prefix=ASM

; IR-NOT: define.*__ehhandler
; IR: define available_externally void @foo(ptr %0)
; IR-NOT: define.*__ehhandler

; No code should be emitted.
; ASM-NOT: __ehtable
; ASM-NOT: __ehhandler

target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc"

declare i32 @__CxxFrameHandler3(...) unnamed_addr

define available_externally void @foo(ptr) personality ptr @__CxxFrameHandler3 {
start:
  invoke void %0()
          to label %good unwind label %bad

good:                                             ; preds = %start
  ret void

bad:                                              ; preds = %start
  %cleanuppad = cleanuppad within none []
  cleanupret from %cleanuppad unwind to caller
}
