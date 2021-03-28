; RUN: llc < %s -enable-shrink-wrap=true | FileCheck %s

; TODO: add preallocated versions of tests
; we don't yet support conditionally called preallocated calls after the setup

; chkstk cannot come before the usual prologue, since it adjusts ESP.
; If chkstk is used in the prologue, we also have to be careful about preserving
; EAX if it is used.

target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc18.0.0"

%struct.S = type { [8192 x i8] }

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
; CHECK: movl $8192, %eax
; CHECK: calll __chkstk
; CHECK: calll _inalloca_params
; CHECK: movl %ebp, %esp
; CHECK: popl %ebp
; CHECK: retl

declare void @inalloca_params(<{ %struct.S }>* inalloca)

declare i32 @doSomething(i32, i32*)

; In this test case, we force usage of EAX before the prologue, and have to
; compensate before calling __chkstk. It would also be valid for us to avoid
; shrink wrapping in this case.

define x86_fastcallcc i32 @use_eax_before_prologue(i32 inreg %a, i32 inreg %b) {
  %tmp = alloca i32, i32 1024, align 4
  %tmp2 = icmp slt i32 %a, %b
  br i1 %tmp2, label %true, label %false

true:
  store i32 %a, i32* %tmp, align 4
  %tmp4 = call i32 @doSomething(i32 0, i32* %tmp)
  br label %false

false:
  %tmp.0 = phi i32 [ %tmp4, %true ], [ %a, %0 ]
  ret i32 %tmp.0
}

; CHECK-LABEL: @use_eax_before_prologue@8: # @use_eax_before_prologue
; CHECK: movl %ecx, %eax
; CHECK: cmpl %edx, %ecx
; CHECK: jge LBB1_2
; CHECK: pushl %eax
; CHECK: movl $4092, %eax
; CHECK: calll __chkstk
; CHECK: movl 4092(%esp), %eax
; CHECK: calll _doSomething
; CHECK: LBB1_2:
; CHECK: retl
