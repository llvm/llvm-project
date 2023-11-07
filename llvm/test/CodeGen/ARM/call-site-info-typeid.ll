; Tests that call site type ids can be extracted and set from type operand
; bundles.

; Verify the exact typeId value to ensure it is not garbage but the value
; computed as the type id from the type operand bundle.
; RUN: llc --call-graph-section -mtriple arm-linux-gnu %s -stop-before=finalize-isel -o - | FileCheck %s

; ModuleID = 'test.c'
source_filename = "test.c"
target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv4t-unknown-linux-gnu"

define dso_local void @foo(i8 signext %a) !type !3 {
entry:
  ret void
}

; CHECK: name: main
define dso_local i32 @main() !type !4 {
entry:
  %retval = alloca i32, align 4
  %fp = alloca void (i8)*, align 8
  store i32 0, i32* %retval, align 4
  store void (i8)* @foo, void (i8)** %fp, align 8
  %0 = load void (i8)*, void (i8)** %fp, align 8
  ; CHECK: callSites:
  ; CHECK-NEXT: - { bb: {{.*}}, offset: {{.*}}, fwdArgRegs: [], typeId:
  ; CHECK-NEXT: 7854600665770582568 }
  call void %0(i8 signext 97) [ "type"(metadata !"_ZTSFvcE.generalized") ]
  ret i32 0
}

!llvm.module.flags = !{!0, !1, !2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"uwtable", i32 1}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i64 0, !"_ZTSFvcE.generalized"}
!4 = !{i64 0, !"_ZTSFiE.generalized"}
