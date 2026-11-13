; RUN: llc -mtriple=aarch64-pc-windows-msvc -enable-shrink-wrap=true \
; RUN:   -verify-machineinstrs %s -o - | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=aarch64-pc-windows-msvc -enable-shrink-wrap=true \
; RUN:   -verify-machineinstrs -filetype=obj %s -o %t.obj
; RUN: llvm-readobj --unwind %t.obj | FileCheck %s --check-prefix=UNWIND

; Forcing shrink wrapping must still honor Windows CFI placement restrictions.
; A supported function moves its frame off the early-return path, whereas a
; personality, signed return address, or realigned stack requires an entry frame.
define i32 @supported(i1 %skip) uwtable {
; ASM-LABEL: supported:
; ASM-NOT: .seh_proc
; ASM: tbz w0, #0,
; ASM: .seh_proc supported
; ASM: .seh_endprologue
; ASM: bl work
; UNWIND: Function: supported +0xC (0xC)
entry:
  br i1 %skip, label %return, label %slow

slow:
  %called = call i32 @work(ptr null)
  %result = add i32 %called, 1
  br label %return

return:
  %value = phi i32 [ 0, %entry ], [ %result, %slow ]
  ret i32 %value
}

define i32 @with_personality(i1 %skip) uwtable personality ptr @__C_specific_handler {
; ASM-LABEL: with_personality:
; ASM: .seh_proc with_personality
; ASM-NOT: tbz
; ASM: .seh_endprologue
; ASM: tbz
; ASM: bl work
; UNWIND: Function: with_personality (
entry:
  br i1 %skip, label %return, label %slow

slow:
  %called = call i32 @work(ptr null)
  %result = add i32 %called, 1
  br label %return

return:
  %value = phi i32 [ 0, %entry ], [ %result, %slow ]
  ret i32 %value
}

define i32 @signed_return(i1 %skip) uwtable "sign-return-address"="non-leaf" "sign-return-address-key"="b_key" {
; ASM-LABEL: signed_return:
; ASM: .seh_proc signed_return
; ASM-NOT: tbz
; ASM: .seh_pac_sign_lr
; ASM: .seh_endprologue
; ASM: tbz
; ASM: bl work
; UNWIND: Function: signed_return (
; UNWIND: 0xfc                ; pacibsp
entry:
  br i1 %skip, label %return, label %slow

slow:
  %called = call i32 @work(ptr null)
  %result = add i32 %called, 1
  br label %return

return:
  %value = phi i32 [ 0, %entry ], [ %result, %slow ]
  ret i32 %value
}

define i32 @realigned_stack(i1 %skip) uwtable {
; ASM-LABEL: realigned_stack:
; ASM: .seh_proc realigned_stack
; ASM-NOT: tbz
; ASM: .seh_endprologue
; ASM: and sp,
; ASM: tbz
; ASM: bl work
; UNWIND: Function: realigned_stack (
entry:
  %buffer = alloca [4 x i64], align 64
  br i1 %skip, label %return, label %slow

slow:
  %called = call i32 @work(ptr %buffer)
  %result = add i32 %called, 1
  br label %return

return:
  %value = phi i32 [ 0, %entry ], [ %result, %slow ]
  ret i32 %value
}

declare i32 @work(ptr)
declare i32 @__C_specific_handler(...)
