; RUN: opt < %s -passes=internalize -S | FileCheck %s
; Internalized symbols should have default visibility.

; CHECK: @global = global i32 0
@global = global i32 0
@llvm.used = appending global [1 x i32*] [i32* @global]

; CHECK: @hidden.variable = internal global i32 0
@hidden.variable = hidden global i32 0
; CHECK: @protected.variable = internal global i32 0
@protected.variable = protected global i32 0

; CHECK: @hidden.alias = internal alias  i32,  i32* @global
@hidden.alias = hidden alias i32, i32* @global
; CHECK: @protected.alias = internal alias i32, i32* @global
@protected.alias = protected alias i32, i32* @global

; CHECK: define internal void @hidden.function() {
define hidden void @hidden.function() {
  ret void
}
; CHECK: define internal void @protected.function() {
define protected void @protected.function() {
  ret void
}
