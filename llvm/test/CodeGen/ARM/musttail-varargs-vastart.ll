; RUN: llc -verify-machineinstrs -mtriple=thumbv7-windows-msvc %s -o - | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=armv7-linux-gnueabi %s -o - | FileCheck %s

;; A variadic function that calls va_start can still forward its arguments
;; with a musttail call. The save area is released before the tail branch.

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

;; A split byval also has a save area. Its register part goes back in r0-r3
;; and its stack part stays where it is.
%struct.S = type { [5 x i32] }
declare void @callee(ptr byval(%struct.S) align 4, ...)

define void @fwd_byval(ptr byval(%struct.S) align 4 %s, ...) nounwind {
; CHECK-LABEL: fwd_byval:
; CHECK:       sub sp, {{(sp, )?}}#16
; CHECK:       stm{{(\.w)?}} sp!, {r0, r1, r2, r3}
; CHECK:       b callee
  musttail call void (ptr, ...) @callee(ptr byval(%struct.S) align 4 %s, ...)
  ret void
}

;; A plain tail call is fine too: it can't use the caller's va_list.
declare void @vcallee(i32, ...)

define void @tail_vastart(i32 %n, ...) nounwind {
; CHECK-LABEL: tail_vastart:
; CHECK:       bl sink
; CHECK:       add sp, {{(sp, )?}}#12
; CHECK-NEXT:  b{{(\.w)?}} vcallee
entry:
  %ap = alloca ptr, align 4
  call void @llvm.va_start(ptr %ap)
  call void @sink(ptr %ap)
  call void @llvm.va_end(ptr %ap)
  tail call void (i32, ...) @vcallee(i32 %n, i32 1)
  ret void
}
