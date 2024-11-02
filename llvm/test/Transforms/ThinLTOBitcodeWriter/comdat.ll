; RUN: opt -thinlto-bc -thinlto-split-lto-unit -o %t %s
; RUN: llvm-modextract -n 0 -o - %t | llvm-dis | FileCheck --check-prefix=THIN %s
; RUN: llvm-modextract -n 1 -o - %t | llvm-dis | FileCheck --check-prefix=MERGED %s

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.0.24215"

; Internal comdat leader with type metadata. All comdat members need to live
; in the merged module, and the comdat needs to be renamed.
; MERGED: ${{"?lwt[^ ]+}} = comdat any
$lwt = comdat any

; External comdat leader, type metadata on non-leader. All comdat
; members need to live in the merged module, internal members need to
; be renamed.
; MERGED: $nlwt = comdat any
$nlwt = comdat any

; Comdat with two members without type metadata. All comdat members live in
; the ThinLTO module and no renaming needs to take place.
; THIN: $nt = comdat any
$nt = comdat any

; MERGED: @lwt_aliasee = private unnamed_addr global
; MERGED-SAME: comdat(${{"?lwt[^ ]+}})
@lwt_aliasee = private unnamed_addr global [1 x ptr] [ptr null], comdat($lwt), !type !0

; MERGED: {{@"?lwt_nl[^ ]+}} = hidden unnamed_addr global
; MERGED-SAME: comdat(${{"?lwt[^ ]+}})
; THIN: {{@"?lwt_nl[^ ]+}} = external hidden
@lwt_nl = internal unnamed_addr global i32 0, comdat($lwt)

; MERGED: @nlwt_aliasee = private unnamed_addr global
; MERGED-SAME: comdat($nlwt)
@nlwt_aliasee = private unnamed_addr global [1 x ptr] [ptr null], comdat($nlwt), !type !0

; MERGED: @nlwt = unnamed_addr global
; MERGED-SAME: comdat
; THIN: @nlwt = external
@nlwt = unnamed_addr global i32 0, comdat

; THIN: @nt = internal
; THIN-SAME: comdat
@nt = internal unnamed_addr global [1 x ptr] [ptr null], comdat

; THIN: @nt_nl = internal
; THIN-SAME: comdat($nt)
@nt_nl = internal unnamed_addr global i32 0, comdat($nt)

; MERGED: {{@"?lwt[^ ]+}} = hidden unnamed_addr alias
; THIN: {{@"?lwt[^ ]+}} = external hidden
@lwt = internal unnamed_addr alias [1 x ptr], ptr @lwt_aliasee

; MERGED: {{@"?nlwt_nl[^ ]+}} = hidden unnamed_addr alias
; THIN: {{@"?nlwt_nl[^ ]+}} = external hidden
@nlwt_nl = internal unnamed_addr alias [1 x ptr], ptr @nlwt_aliasee

; The functions below exist just to make sure the globals are used.
define ptr @lwt_fun() {
  %1 = load i32, ptr @lwt_nl
  %2 = getelementptr inbounds [1 x ptr], ptr @lwt, i32 0, i32 %1
  %3 = load ptr, ptr %2
  ret ptr %3
}

define ptr @nlwt_fun() {
  %1 = load i32, ptr @nlwt
  %2 = getelementptr inbounds [1 x ptr], ptr @nlwt_nl, i32 0, i32 %1
  %3 = load ptr, ptr %2
  ret ptr %3
}

define ptr @nt_fun() {
  %1 = load i32, ptr @nt_nl
  %2 = getelementptr inbounds [1 x ptr], ptr @nt, i32 0, i32 %1
  %3 = load ptr, ptr %2
  ret ptr %3
}

!0 = !{i64 8, !"?AVA@@"}
