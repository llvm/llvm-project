; RUN: llc < %s -march=nvptx64 -mcpu=sm_32 | FileCheck %s --check-prefixes=SM30,CHECK
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_32 | %ptxas-verify %}
; RUN: llc < %s -march=nvptx64 -mcpu=sm_70 -mattr=+ptx63 | FileCheck %s --check-prefixes=SM70,CHECK
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_70 -mattr=+ptx63 | %ptxas-verify -arch=sm_70 %}

; TODO: these are system scope, but are compiled to gpu scope..
; TODO: these are seq_cst, but are compiled to relaxed..

; CHECK-LABEL: relaxed_sys_i8
define i8 @relaxed_sys_i8(ptr %addr, i8 %cmp, i8 %new) {
  ; SM30: atom.cas.b32
  ; SM70: atom.cas.b16
  %pairold = cmpxchg ptr %addr, i8 %cmp, i8 %new seq_cst seq_cst
  ret i8 %new
}

; CHECK-LABEL: relaxed_sys_i16
define i16 @relaxed_sys_i16(ptr %addr, i16 %cmp, i16 %new) {
  ; SM30: atom.cas.b32
  ; SM70: atom.cas.b16
  %pairold = cmpxchg ptr %addr, i16 %cmp, i16 %new seq_cst seq_cst
  ret i16 %new
}

; CHECK-LABEL: relaxed_sys_i32
define i32 @relaxed_sys_i32(ptr %addr, i32 %cmp, i32 %new) {
  ; CHECK: atom.cas.b32
  %pairold = cmpxchg ptr %addr, i32 %cmp, i32 %new seq_cst seq_cst
  ret i32 %new
}

; CHECK-LABEL: relaxed_sys_i64
define i64 @relaxed_sys_i64(ptr %addr, i64 %cmp, i64 %new) {
  ; CHECK: atom.cas.b64
  %pairold = cmpxchg ptr %addr, i64 %cmp, i64 %new seq_cst seq_cst
  ret i64 %new
}
