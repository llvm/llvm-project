; RUN: rm -rf %t && split-file %s %t
; RUN: llc -verify-machineinstrs -mtriple=thumbv7-windows-msvc %t/tail.ll -o - | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=armv7-linux-gnueabi %t/tail.ll -o - | FileCheck %s
; RUN: not --crash llc -mtriple=thumbv8m.base-none-eabi %t/thumb1.ll -o - 2>&1 | FileCheck %s --check-prefix=THUMB1

;; A variadic function that calls va_start can still forward its arguments
;; with a musttail call. The save area is released before the tail branch.

;--- tail.ll
declare void @llvm.va_start(ptr)
declare void @llvm.va_end(ptr)
declare void @sink(ptr)

define void @fwd_vastart(ptr %pfn, ...) nounwind {
; CHECK-LABEL: fwd_vastart:
; CHECK:       sub sp, {{(sp, )?}}#12
; CHECK:       stm r0{{!?}}, {r1, r2, r3}
; CHECK:       bl sink
; CHECK:       mov r1, r{{[0-9]+}}
; CHECK-NEXT:  mov r2, r{{[0-9]+}}
; CHECK-NEXT:  mov r3, r{{[0-9]+}}
; CHECK:       pop{{(\.w)?}} {
; CHECK:       add sp, {{(sp, )?}}#12
; CHECK:       bx r{{[0-9]+$}}
entry:
  %ap = alloca ptr, align 4
  call void @llvm.va_start(ptr %ap)
  call void @sink(ptr %ap)
  call void @llvm.va_end(ptr %ap)
  musttail call void (ptr, ...) %pfn(ptr %pfn, ...)
  ret void
}

define void @fwd_vastart_named1(ptr %pfn, i32 %n, ...) nounwind {
; CHECK-LABEL: fwd_vastart_named1:
; CHECK:       sub sp, {{(sp, )?}}#8
; CHECK:       bl sink
; CHECK:       pop{{(\.w)?}} {
; CHECK:       add sp, {{(sp, )?}}#8
; CHECK:       bx r{{[0-9]+$}}
entry:
  %ap = alloca ptr, align 4
  call void @llvm.va_start(ptr %ap)
  call void @sink(ptr %ap)
  call void @llvm.va_end(ptr %ap)
  musttail call void (ptr, i32, ...) %pfn(ptr %pfn, i32 %n, ...)
  ret void
}

;--- thumb1.ll
;; Thumb1 is not handled and still fails.
declare void @llvm.va_start(ptr)
declare void @llvm.va_end(ptr)
declare void @sink(ptr)

define void @fwd_vastart_thumb1(ptr %pfn, ...) nounwind {
entry:
  %ap = alloca ptr, align 4
  call void @llvm.va_start(ptr %ap)
  call void @sink(ptr %ap)
  call void @llvm.va_end(ptr %ap)
  musttail call void (ptr, ...) %pfn(ptr %pfn, ...)
  ret void
}

; THUMB1: LLVM ERROR: failed to perform tail call elimination on a call site marked musttail
