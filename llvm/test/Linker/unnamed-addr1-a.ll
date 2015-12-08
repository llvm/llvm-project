; RUN: llvm-link %s %p/unnamed-addr1-b.ll -S -o - | FileCheck %s

; Only in this file
@global-a = common global i32 0
; CHECK-DAG: @global-a = common global i32 0
@global-b = common unnamed_addr global i32 0
; CHECK-DAG: @global-b = common unnamed_addr global i32 0

define weak void @func-a() { ret void }
; CHECK-DAG: define weak void @func-a() {
define weak void @func-b() unnamed_addr { ret void }
; CHECK-DAG: define weak void @func-b() unnamed_addr {

; Other file has unnamed_addr definition
@global-c = common unnamed_addr global i32 0
; CHECK-DAG: @global-c = common unnamed_addr global i32 0
@global-d = external global i32

define i32* @use-global-d() {
  ret i32* @global-d
}

; CHECK-DAG: @global-d = global i32 42
@global-e = external unnamed_addr global i32
; CHECK-DAG: @global-e = unnamed_addr global i32 42
@global-f = weak global i32 42
; CHECK-DAG: @global-f = global i32 42

@alias-a = weak global i32 42
; CHECK-DAG: @alias-a = alias i32, i32* @global-f
@alias-b = weak unnamed_addr global i32 42
; CHECK-DAG: @alias-b = unnamed_addr alias i32, i32* @global-f

declare void @func-c()
define void @use-func-c() {
  call void @func-c()
  ret void
}

; CHECK-DAG: define weak void @func-c() {
define weak void @func-d() { ret void }
; CHECK-DAG: define weak void @func-d() {
define weak void @func-e() unnamed_addr { ret void }
; CHECK-DAG: define weak void @func-e() unnamed_addr {

; Other file has non-unnamed_addr definition
@global-g = common unnamed_addr global i32 0
; CHECK-DAG: @global-g = common global i32 0
@global-h = external global i32
; CHECK-DAG: @global-h = global i32 42
@global-i = external unnamed_addr global i32
; CHECK-DAG: @global-i = global i32 42
@global-j = weak global i32 42
; CHECK-DAG: @global-j = global i32 42

@alias-c = weak global i32 42
; CHECK-DAG: @alias-c = alias i32, i32* @global-f
@alias-d = weak unnamed_addr global i32 42
; CHECK-DAG: @alias-d = alias i32, i32* @global-f


declare void @func-g()
; CHECK-DAG: define weak void @func-g() {
define weak void @func-h() { ret void }
; CHECK-DAG: define weak void @func-h() {
define weak void @func-i() unnamed_addr { ret void }
; CHECK-DAG: define weak void @func-i() {
