; RUN: opt < %s -dfsan -dfsan-args-abi -dfsan-abilist=%S/Inputs/abilist.txt -S | FileCheck %s
; RUN: opt < %s -dfsan                 -dfsan-abilist=%S/Inputs/abilist.txt -S | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

define internal i8 @uninstrumented_internal_fun(i8 %in) {
  ret i8 %in
}

define i8 @call_uninstrumented_internal_fun(i8 %in) {
  %call = call i8 @uninstrumented_internal_fun(i8 %in)
  ret i8 %call
}
; CHECK: define internal {{(i8|{ i8, i16 })}} @"dfsw$uninstrumented_internal_fun"

define private i8 @uninstrumented_private_fun(i8 %in) {
  ret i8 %in
}

define i8 @call_uninstrumented_private_fun(i8 %in) {
  %call = call i8 @uninstrumented_private_fun(i8 %in)
  ret i8 %call
}
; CHECK: define private {{(i8|{ i8, i16 })}} @"dfsw$uninstrumented_private_fun"
