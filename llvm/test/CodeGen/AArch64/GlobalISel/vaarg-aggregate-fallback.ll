; RUN: not llc %s -mtriple=aarch64-- -O0 -global-isel -global-isel-abort=2 \
; RUN:   -pass-remarks-missed='gisel*' -filetype=null 2>&1 | FileCheck %s

; The code generator does not support va_arg with an aggregate type. The
; IRTranslator gives up on it, so that the fallback to SelectionDAG reports it
; instead of crashing (see issue #162900).

; CHECK: remark: {{.*}}unable to translate instruction: va_arg:{{.*}}(in function: struct)
; CHECK: warning: Instruction selection used fallback path for struct
; CHECK: error: {{.*}}in function struct {{.*}}: va_arg with an aggregate type is not supported
define { i32, i64 } @struct(ptr %ap) {
  %v = va_arg ptr %ap, { i32, i64 }
  ret { i32, i64 } %v
}
