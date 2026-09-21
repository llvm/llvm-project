; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

; No target can lower va_arg with an aggregate type, so frontends expand it
; into accesses to its members instead (see issue #162900).

; CHECK: va_arg with an aggregate type is not supported
; CHECK-NEXT: %s = va_arg ptr %ap, { i32, i64 }
define { i32, i64 } @struct(ptr %ap) {
  %s = va_arg ptr %ap, { i32, i64 }
  ret { i32, i64 } %s
}

; CHECK: va_arg with an aggregate type is not supported
; CHECK-NEXT: %a = va_arg ptr %ap, [2 x i32]
define [2 x i32] @array(ptr %ap) {
  %a = va_arg ptr %ap, [2 x i32]
  ret [2 x i32] %a
}
