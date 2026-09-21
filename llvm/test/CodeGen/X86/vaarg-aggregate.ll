; RUN: not llc %s -mtriple=x86_64-- -filetype=null 2>&1 | FileCheck %s
; RUN: not llc %s -mtriple=x86_64-- -O0 -filetype=null 2>&1 | FileCheck %s
; RUN: not llc %s -mtriple=i686-- -filetype=null 2>&1 | FileCheck %s

; The code generator does not support va_arg with an aggregate type on any
; target. Emit a clean diagnostic instead of crashing (see issue #162900).

; CHECK: error: {{.*}}in function packed_struct {{.*}}: va_arg with an aggregate type is not supported
define <{ i32 }> @packed_struct(ptr %ap) {
  %v = va_arg ptr %ap, <{ i32 }>
  ret <{ i32 }> %v
}

; CHECK: error: {{.*}}in function struct {{.*}}: va_arg with an aggregate type is not supported
define { i32, i64 } @struct(ptr %ap) {
  %v = va_arg ptr %ap, { i32, i64 }
  ret { i32, i64 } %v
}

; CHECK: error: {{.*}}in function array {{.*}}: va_arg with an aggregate type is not supported
define [2 x i32] @array(ptr %ap) {
  %v = va_arg ptr %ap, [2 x i32]
  ret [2 x i32] %v
}
