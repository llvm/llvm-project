; RUN: llc < %s -march=x86 -relocation-model=pic -mtriple=i686-unknown-linux-gnu -fast-isel | FileCheck %s
; PR3654

@v = thread_local global i32 0
define i32 @f() nounwind {
entry:
          %t = load i32* @v
          %s = add i32 %t, 1
          ret i32 %s
}

; CHECK: f:
; CHECK: leal	v@TLSGD
; CHECK: __tls_get_addr

@alias = alias internal i32* @v
define i32 @f_alias() nounwind {
entry:
          %t = load i32* @v
          %s = add i32 %t, 1
          ret i32 %s
}

; CHECK: f_alias:
; CHECK: leal	v@TLSGD
; CHECK: __tls_get_addr
