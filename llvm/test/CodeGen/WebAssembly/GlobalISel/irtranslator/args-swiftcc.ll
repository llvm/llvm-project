; RUN: llc -mtriple=wasm32 -global-isel -stop-after=irtranslator -verify-machineinstrs < %s | FileCheck %s -check-prefixes=CHECK,WASM32
; RUN: llc -mtriple=wasm64 -global-isel -stop-after=irtranslator -verify-machineinstrs < %s | FileCheck %s -check-prefixes=CHECK,WASM64

define swiftcc void @test_implicit_self_and_error(float %arg) {
  ; CHECK-LABEL: name: test_implicit_self_and_error

  ; CHECK: machineFunctionInfo:
  ; WASM32-NEXT:   params: [ f32, i32, i32 ]
  ; WASM64-NEXT:   params: [ f32, i64, i64 ]
  ; CHECK-NEXT:   results: [ ]

  ; CHECK: bb.1 (%ir-block.0):
  ; CHECK-NEXT:   liveins: $arguments
  ; CHECK-NEXT: {{  $}}
  ; CHECK-NEXT:   [[ARGUMENT_f32_:%[0-9]+]]:f32(s32) = ARGUMENT_f32 0, implicit $arguments
  ret void
}

define swiftcc void @test_explicit_self_and_implicit_error(ptr swiftself %self, float %arg) {
  ; CHECK-LABEL: name: test_explicit_self_and_implicit_error


  ; CHECK: machineFunctionInfo:
  ; WASM32-NEXT:   params: [ i32, f32, i32 ]
  ; WASM64-NEXT:   params: [ i64, f32, i64 ]
  ; CHECK-NEXT:   results: [ ]

  ; CHECK: bb.1 (%ir-block.0):
  ; CHECK-NEXT:   liveins: $arguments
  ; CHECK-NEXT: {{  $}}
  ; WASM32-NEXT:   [[ARGUMENT_i32_:%[0-9]+]]:i32(p0) = ARGUMENT_i32 0, implicit $arguments
  ; WASM64-NEXT:   [[ARGUMENT_i64_:%[0-9]+]]:i64(p0) = ARGUMENT_i64 0, implicit $arguments
  ; CHECK-NEXT:   [[ARGUMENT_f32_:%[0-9]+]]:f32(s32) = ARGUMENT_f32 1, implicit $arguments
  ret void
}

define swiftcc void @test_implicit_self_and_explicit_error(ptr swifterror %error, float %arg) {
  ; CHECK-LABEL: name: test_implicit_self_and_explicit_error


  ; CHECK: machineFunctionInfo:
  ; WASM32-NEXT:   params: [ i32, f32, i32 ]
  ; WASM64-NEXT:   params: [ i64, f32, i64 ]
  ; CHECK-NEXT:   results: [ ]

  ; CHECK: bb.1 (%ir-block.0):
  ; CHECK-NEXT:   liveins: $arguments
  ; CHECK-NEXT: {{  $}}
  ; WASM32-NEXT:   [[ARGUMENT_i32_:%[0-9]+]]:i32(p0) = ARGUMENT_i32 0, implicit $arguments
  ; WASM64-NEXT:   [[ARGUMENT_i64_:%[0-9]+]]:i64(p0) = ARGUMENT_i64 0, implicit $arguments
  ; CHECK-NEXT:   [[ARGUMENT_f32_:%[0-9]+]]:f32(s32) = ARGUMENT_f32 1, implicit $arguments
  ret void
}

define swiftcc void @test_explicit_self_and_error(ptr swiftself %self, ptr swifterror %error, float %arg) {
  ; CHECK-LABEL: name: test_explicit_self_and_error


  ; CHECK: machineFunctionInfo:
  ; WASM32-NEXT:   params: [ i32, i32, f32 ]
  ; WASM64-NEXT:   params: [ i64, i64, f32 ]
  ; CHECK-NEXT:   results: [ ]

  ; CHECK: bb.1 (%ir-block.0):
  ; CHECK-NEXT:   liveins: $arguments
  ; CHECK-NEXT: {{  $}}
  ; WASM32-NEXT:   [[ARGUMENT_i32_:%[0-9]+]]:i32(p0) = ARGUMENT_i32 0, implicit $arguments
  ; WASM32-NEXT:   [[ARGUMENT_i32_1:%[0-9]+]]:i32(p0) = ARGUMENT_i32 1, implicit $arguments
  ; WASM64-NEXT:   [[ARGUMENT_i64_:%[0-9]+]]:i64(p0) = ARGUMENT_i64 0, implicit $arguments
  ; WASM64-NEXT:   [[ARGUMENT_i64_1:%[0-9]+]]:i64(p0) = ARGUMENT_i64 1, implicit $arguments
  ; CHECK-NEXT:   [[ARGUMENT_f32_:%[0-9]+]]:f32(s32) = ARGUMENT_f32 2, implicit $arguments
  ret void
}
