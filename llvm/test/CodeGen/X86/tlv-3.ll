; RUN: llc < %s -mtriple x86_64-apple-darwin | FileCheck %s
; PR17964

; CHECK: __DATA,__thread_data,thread_local_regular
; CHECK: _foo$tlv$init
@foo = weak_odr thread_local global i8 1, align 4

define i32 @main() {
    ret i32 0
}
