; RUN: llc -mtriple aarch64-unknown-windows-msvc -filetype asm -o - %s | FileCheck %s -check-prefixes=CHECK,DAG-ISEL
; RUN: llc -mtriple aarch64-unknown-windows-msvc -fast-isel -filetype asm -o - %s | FileCheck %s -check-prefixes=CHECK,FAST-ISEL
; RUN: llc -mtriple aarch64-unknown-windows-msvc -verify-machineinstrs -O0 -filetype asm -o - %s | FileCheck %s -check-prefixes=CHECK,GLOBAL-ISEL,GLOBAL-ISEL-FALLBACK

@var = external dllimport global i32
@ext = external global i32
declare dllimport i32 @external()
declare i32 @internal()

define i32 @get_var() {
  %1 = load i32, ptr @var, align 4
  ret i32 %1
}

; CHECK-LABEL: get_var
; CHECK: adrp x8, __imp_var
; CHECK: ldr x8, [x8, :lo12:__imp_var]
; CHECK: ldr w0, [x8]
; CHECK: ret

define i32 @get_ext() {
  %1 = load i32, ptr @ext, align 4
  ret i32 %1
}

; CHECK-LABEL: get_ext
; CHECK: adrp x8, ext
; DAG-ISEL: ldr w0, [x8, :lo12:ext]
; FAST-ISEL: add x8, x8, :lo12:ext
; FAST-ISEL: ldr w0, [x8]
; GLOBAL-ISEL-FALLBACK: ldr w0, [x8, :lo12:ext]
; CHECK: ret

define ptr @get_var_pointer() {
  ret ptr @var
}

; CHECK-LABEL: get_var_pointer
; CHECK: adrp [[REG1:x[0-9]+]], __imp_var
; CHECK: ldr {{x[0-9]+}}, [[[REG1]], :lo12:__imp_var]
; CHECK: ret

define i32 @call_external() {
  %call = tail call i32 @external()
  ret i32 %call
}

; CHECK-LABEL: call_external
; CHECK: adrp x0, __imp_external
; CHECK: ldr x0, [x0, :lo12:__imp_external]
; CHECK: br x0

define i32 @call_internal() {
  %call = tail call i32 @internal()
  ret i32 %call
}

; CHECK-LABEL: call_internal
; DAG-ISEL: b internal
; FAST-ISEL: b internal
; GLOBAL-ISEL: b internal

define void @call_try_catch() personality ptr @__gxx_personality_seh0 {
entry:
  invoke void @myFunc()
          to label %try.cont unwind label %lpad

lpad:                                             ; preds = %entry
  %0 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %1 = extractvalue { ptr, i32 } %0, 1
  %2 = tail call i32 @llvm.eh.typeid.for(ptr nonnull @_ZTIi) #1
  %matches = icmp eq i32 %1, %2
  br i1 %matches, label %catch, label %eh.resume

catch:                                            ; preds = %lpad
  %3 = extractvalue { ptr, i32 } %0, 0
  %4 = tail call ptr @__cxa_begin_catch(ptr %3) #1
  tail call void @__cxa_end_catch() #1
  br label %try.cont

try.cont:                                         ; preds = %entry, %catch
  ret void

eh.resume:                                        ; preds = %lpad
  resume { ptr, i32 } %0
}

; CHECK-LABEL: call_try_catch
; CHECK: adrp x8, __imp_myFunc
; CHECK: ldr x8, [x8, :lo12:__imp_myFunc]
; CHECK: blr x8

declare dllimport void @myFunc()

@_ZTIi = external constant ptr
declare dso_local i32 @__gxx_personality_seh0(...)
declare i32 @llvm.eh.typeid.for(ptr) #0
declare dso_local ptr @__cxa_begin_catch(ptr)
declare dso_local void @__cxa_end_catch()

attributes #0 = { nounwind memory(none) }
attributes #1 = { nounwind }
