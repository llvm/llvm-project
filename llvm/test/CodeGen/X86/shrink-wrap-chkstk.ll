; RUN: llc < %s -enable-shrink-wrap=true | FileCheck %s

; chkstk cannot come before the usual prologue, since it adjusts ESP.

target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc18.0.0"

%struct.S = type { [12 x i8] }

define x86_thiscallcc void @call_inalloca(i1 %x) {
entry:
  %argmem = alloca inalloca <{ %struct.S }>, align 4
  %argidx1 = getelementptr inbounds <{ %struct.S }>, <{ %struct.S }>* %argmem, i32 0, i32 0, i32 0, i32 0
  %argidx2 = getelementptr inbounds <{ %struct.S }>, <{ %struct.S }>* %argmem, i32 0, i32 0, i32 0, i32 1
  store i8 42, i8* %argidx2, align 4
  br i1 %x, label %bb1, label %bb2

bb1:
  store i8 42, i8* %argidx1, align 4
  br label %bb2

bb2:
  call void @inalloca_params(<{ %struct.S }>* inalloca nonnull %argmem)
  ret void
}

; CHECK-LABEL: _call_inalloca: # @call_inalloca
; CHECK: pushl %ebp
; CHECK: movl %esp, %ebp
; CHECK: movl $12, %eax
; CHECK: calll __chkstk
; CHECK: calll _inalloca_params
; CHECK: movl %ebp, %esp
; CHECK: popl %ebp
; CHECK: retl

declare void @inalloca_params(<{ %struct.S }>* inalloca)
