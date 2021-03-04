; RUN: opt < %s -dfsan -dfsan-args-abi -dfsan-abilist=%S/Inputs/abilist.txt -S | FileCheck %s --check-prefixes=CHECK,ARGS_ABI
; RUN: opt < %s -dfsan                 -dfsan-abilist=%S/Inputs/abilist.txt -S | FileCheck %s --check-prefixes=CHECK,TLS_ABI

; CHECK: @__dfsan_shadow_width_bits = weak_odr constant i32 [[#SBITS:]]
; CHECK: @__dfsan_shadow_width_bytes = weak_odr constant i32 [[#SBYTES:]]

target triple = "x86_64-unknown-linux-gnu"

define internal i8 @uninstrumented_internal_fun(i8 %in) {
  ret i8 %in
}

define i8 @call_uninstrumented_internal_fun(i8 %in) {
  %call = call i8 @uninstrumented_internal_fun(i8 %in)
  ret i8 %call
}
; TLS_ABI: define internal i8 @"dfsw$uninstrumented_internal_fun"
; ARGS_ABI: define internal { i8, i[[#SBITS]] } @"dfsw$uninstrumented_internal_fun"

define private i8 @uninstrumented_private_fun(i8 %in) {
  ret i8 %in
}

define i8 @call_uninstrumented_private_fun(i8 %in) {
  %call = call i8 @uninstrumented_private_fun(i8 %in)
  ret i8 %call
}
; TLS_ABI: define private i8 @"dfsw$uninstrumented_private_fun"
; ARGS_ABI: define private { i8, i[[#SBITS]] } @"dfsw$uninstrumented_private_fun"
