; RUN: llc -global-isel < %s | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

@.str.2 = private unnamed_addr constant [7 x i8] c"Boom!\0A\00", align 1

define dso_local void @trap() {
entry:
  unreachable
}

define dso_local void @test() personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:

; CHECK-LABEL: test:
; CHECK: .Ltmp0:
; CHECK: bl trap
; CHECK: .Ltmp1:

  invoke void asm sideeffect unwind "bl trap", ""()
          to label %invoke.cont unwind label %lpad

invoke.cont:
  ret void

lpad:
  %0 = landingpad { i8*, i32 }
          cleanup
; CHECK: bl	printf
  call void (i8*, ...) @printf(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.2, i64 0, i64 0))
  resume { i8*, i32 } %0

}

declare dso_local i32 @__gxx_personality_v0(...)

declare dso_local void @printf(i8*, ...)

; Exception table generation around the inline assembly

; CHECK-LABEL: GCC_except_table1:
; CHECK-NEXT: .Lexception0:
; CHECK-NEXT: 	.byte	255                             // @LPStart Encoding = omit
; CHECK-NEXT: 	.byte	255                             // @TType Encoding = omit
; CHECK-NEXT: 	.byte	1                               // Call site Encoding = uleb128
; CHECK-NEXT: 	.uleb128 .Lcst_end0-.Lcst_begin0
; CHECK-NEXT: .Lcst_begin0:
; CHECK-NEXT: 	.uleb128 .Ltmp0-.Lfunc_begin0           // >> Call Site 1 <<
; CHECK-NEXT: 	.uleb128 .Ltmp1-.Ltmp0                  //   Call between .Ltmp0 and .Ltmp1
; CHECK-NEXT: 	.uleb128 .Ltmp2-.Lfunc_begin0           //     jumps to .Ltmp2
; CHECK-NEXT: 	.byte	0                               //   On action: cleanup
; CHECK-NEXT: 	.uleb128 .Ltmp1-.Lfunc_begin0           // >> Call Site 2 <<
; CHECK-NEXT: 	.uleb128 .Lfunc_end1-.Ltmp1             //   Call between .Ltmp1 and .Lfunc_end1
; CHECK-NEXT: 	.byte	0                               //     has no landing pad
; CHECK-NEXT: 	.byte	0                               //   On action: cleanup
; CHECK-NEXT: .Lcst_end0: