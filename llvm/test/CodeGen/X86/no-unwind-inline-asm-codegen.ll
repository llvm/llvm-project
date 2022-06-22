; RUN: llc < %s | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str.2 = private unnamed_addr constant [7 x i8] c"Boom!\0A\00", align 1

define dso_local void @trap() {
entry:
  unreachable
}

define dso_local void @test() personality ptr @__gxx_personality_v0 {
entry:

; CHECK-NOT: .Ltmp0:
; CHECK: callq  trap
; CHECK-NOT: .Ltmp1:

  invoke void asm sideeffect "call trap", "~{dirflag},~{fpsr},~{flags}"()
          to label %invoke.cont unwind label %lpad

invoke.cont:
  ret void

lpad:
  %0 = landingpad { ptr, i32 }
          cleanup
  call void (ptr, ...) @printf(ptr @.str.2)
  resume { ptr, i32 } %0

}

declare dso_local i32 @__gxx_personality_v0(...)

declare dso_local void @printf(ptr, ...)

; Exception table generation around the inline assembly shouldn't exist

; CHECK-LABEL: GCC_except_table1:
; CHECK-NEXT: .Lexception0:
; CHECK-NEXT:  .byte 255                             # @LPStart Encoding = omit
; CHECK-NEXT:  .byte 255                             # @TType Encoding = omit
; CHECK-NEXT:  .byte 1                               # Call site Encoding = uleb128
; CHECK-NEXT:  .uleb128 .Lcst_end0-.Lcst_begin0
; CHECK-NEXT: .Lcst_begin0:
; CHECK-NEXT:   .uleb128 .Lfunc_begin0-.Lfunc_begin0    # >> Call Site 1 <<
; CHECK-NEXT:   .uleb128 .Lfunc_end1-.Lfunc_begin0      #   Call between .Lfunc_begin0 and .Lfunc_end1
; CHECK-NEXT:   .byte 0                                 #     has no landing pad
; CHECK-NEXT:   .byte 0                                 #   On action: cleanup
; CHECK-NEXT: .Lcst_end0:
