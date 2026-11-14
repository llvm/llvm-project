; RUN: opt -S -passes=sjlj-eh-prepare -mtriple=i686-linux-gnu < %s | FileCheck -check-prefix=P32 %s
; RUN: opt -S -passes=sjlj-eh-prepare -mtriple=x86_64-linux-gnu < %s | FileCheck -check-prefix=P64 %s
; RUN: opt -S -passes=sjlj-eh-prepare -mtriple=x86_64-linux-gnux32 -data-layout=p:32:32 < %s | FileCheck -check-prefix=X32 %s
; RUN: opt -S -passes=sjlj-eh-prepare -data-layout=p:32:32 < %s | FileCheck -check-prefix=P32 %s

; The function context matches libgcc's SjLj_Function_Context: call_site is
; always i32, and the __data words are register sized.

; P32: %fn_context = alloca { ptr, i32, [4 x i32], ptr, ptr, [5 x ptr] }
; P32: store volatile i32 1, ptr %call_site
; P32: %exn_val = load volatile i32, ptr %exception_gep
; P32: %exn_selector_val = load volatile i32, ptr %exn_selector_gep

; P64: %fn_context = alloca { ptr, i32, [4 x i64], ptr, ptr, [5 x ptr] }
; P64: store volatile i32 1, ptr %call_site
; P64: %exn_val = load volatile i64, ptr %exception_gep
; P64: %exn_selector_val = load volatile i64, ptr %exn_selector_gep

; X32: %fn_context = alloca { ptr, i32, [4 x i64], ptr, ptr, [5 x ptr] }
; X32: store volatile i32 1, ptr %call_site
; X32: %exn_val = load volatile i64, ptr %exception_gep
; X32: %exn_selector_val = load volatile i64, ptr %exn_selector_gep
define void @test() personality ptr @__gxx_personality_sj0 {
entry:
  invoke void @may_throw()
          to label %cont unwind label %lpad

cont:
  ret void

lpad:
  %lp = landingpad { ptr, i32 }
          cleanup
  resume { ptr, i32 } %lp
}

declare void @may_throw()
declare i32 @__gxx_personality_sj0(...)

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"exception-model", !"sjlj"}
