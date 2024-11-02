; Check that we don't replace uses in cmp with wrapper (which would accidentally optimize out the cmp).
; RUN: opt < %s -passes=dfsan -dfsan-abilist=%S/Inputs/abilist.txt -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @__dfsan_shadow_width_bits = weak_odr constant i32 [[#SBITS:]]
; CHECK: @__dfsan_shadow_width_bytes = weak_odr constant i32 [[#SBYTES:]]

; CHECK: declare extern_weak i8 @ExternWeak(i8)
declare extern_weak i8 @ExternWeak(i8)

define noundef i8 @call_if_exists() local_unnamed_addr {
  ; CHECK-LABEL: @call_if_exists.dfsan
  ; Ensure comparison is preserved
  ; CHECK: br i1 icmp ne ([[FUNCPTRTY:.*]] @ExternWeak, [[FUNCPTRTY]] null), label %use_func, label %avoid_func
  br i1 icmp ne (ptr @ExternWeak, ptr null), label %use_func, label %avoid_func

use_func:
  ; CHECK: use_func:
  ; Ensure extern weak function is validated before being called.
  ; CHECK: call void @__dfsan_wrapper_extern_weak_null({{[^,]*}}@ExternWeak{{[^,]*}}, {{.*}})
  ; CHECK-NEXT: call i8 @ExternWeak(i8 {{.*}})
  %1 = call i8 @ExternWeak(i8 4)
  br label %end

avoid_func:
  ; CHECK: avoid_func:
  br label %end

end:
  ; CHECK: end:
  %r = phi i8 [ %1, %use_func ], [ 0, %avoid_func ]
  ret i8 %r
}

