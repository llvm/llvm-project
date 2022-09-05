; RUN: not opt -verify %s 2>&1 | FileCheck %s

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-gnu"

; CHECK:      dllexport GlobalValue must have default visibility
; CHECK-NEXT: ptr @dllexport_hidden
declare hidden dllexport i32 @dllexport_hidden()
; CHECK:      dllexport GlobalValue must have default visibility
; CHECK-NEXT: ptr @dllexport_protected
declare protected dllexport i32 @dllexport_protected()

; CHECK:      dllimport GlobalValue must have default visibility
; CHECK-NEXT: ptr @dllimport_hidden
declare hidden dllimport i32 @dllimport_hidden()
; CHECK:      dllimport GlobalValue must have default visibility
; CHECK-NEXT: ptr @dllimport_protected
declare protected dllimport i32 @dllimport_protected()
