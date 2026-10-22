; RUN: opt -passes=assign-guid,wholeprogramdevirt -devirtualize-speculatively -wholeprogramdevirt-max-speculative-targets=2 -wholeprogramdevirt-summary-action=export \
; RUN:  -wholeprogramdevirt-read-summary=%S/Inputs/export-speculative.yaml -wholeprogramdevirt-write-summary=%t -S -o - %s
; RUN: FileCheck %s < %t

; Test that WPD resolutions are properly exported for speculative devirt resolutions.

; CHECK:      WPDRes:
; CHECK-NEXT:     0:
; CHECK-NEXT:     Kind:            Speculative
; CHECK-NEXT:     SingleImplName:  ''
; CHECK-NEXT:     ResByArg:        {}
; CHECK-NEXT:     SpeculatedFnNames:
; CHECK-NEXT:         - _ZN1A1fEv
; CHECK-NEXT:         - _ZN1B1fEv

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

@_ZTV1A = constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr null, ptr @_ZN1A1fEv] }, !type !0
@_ZTV1B = constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr null, ptr @_ZN1B1fEv] }, !type !0

declare i32 @_ZN1A1fEv(ptr %this)
declare i32 @_ZN1B1fEv(ptr %this)

!0 = !{i64 16, !"_ZTS1A"}
