; RUN: opt < %s -passes=dfsan -S --dfsan-abilist=%S/Inputs/personality-routine-abilist.txt | FileCheck %s
; RUN: opt < %s -passes=dfsan -S --dfsan-abilist=%S/Inputs/personality-routine-abilist.txt -dfsan-ignore-personality-routine | FileCheck %s --check-prefix=CHECK-IGNORE

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare i32 @__gxx_personality_v0(...)

declare ptr @__cxa_begin_catch(ptr)

declare void @__cxa_end_catch()

declare void @g(...)

; CHECK-LABEL: @h.dfsan
; CHECK-SAME: personality {{.*}}@"dfsw$__gxx_personality_v0"{{.*}}
; CHECK-IGNORE-LABEL: @h.dfsan
; CHECK-IGNORE-SAME: personality {{.*}}__gxx_personality_v0{{.*}}
define i32 @h() personality ptr @__gxx_personality_v0 {
  invoke void (...) @g(i32 42)
          to label %try.cont unwind label %lpad

lpad:
  %1 = landingpad { ptr, i32 }
          catch ptr null
  %2 = extractvalue { ptr, i32 } %1, 0
  %3 = tail call ptr @__cxa_begin_catch(ptr %2)
  tail call void @__cxa_end_catch()
  br label %try.cont

try.cont:
  ret i32 0
}

; CHECK: @"dfsw$__gxx_personality_v0"
; CHECK: call void @__dfsan_vararg_wrapper
; CHECK-IGNORE-NOT: @"dfsw$__gxx_personality_v0"
; CHECK-IGNORE-NOT: call void @__dfsan_vararg_wrapper
