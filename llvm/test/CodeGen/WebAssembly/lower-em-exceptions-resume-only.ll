; RUN: opt < %s -wasm-lower-em-ehsjlj -enable-emscripten-cxx-exceptions -S | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; Checks if a module that only contains a resume but not an invoke works
; correctly and does not crash.
; CHECK-LABEL: @resume_only
; CHECK: call void @__resumeException
define void @resume_only() personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %val0 = insertvalue { i8*, i32 } undef, i8* null, 0
  %val1 = insertvalue { i8*, i32} %val0, i32 0, 1
  resume { i8*, i32 } %val1
}

declare i32 @__gxx_personality_v0(...)
