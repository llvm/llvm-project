; RUN: llc -mtriple=i686-pc-windows-msvc < %s | FileCheck %s

declare void @may_throw_or_crash()
declare i32 @_except_handler3(...)
declare i32 @_except_handler4(...)
declare i32 @__CxxFrameHandler3(...)
declare void @llvm.eh.begincatch(i8*, i8*)
declare void @llvm.eh.endcatch()
declare i32 @llvm.eh.typeid.for(i8*)

define internal i32 @catchall_filt() {
  ret i32 1
}

define void @use_except_handler3() personality i32 (...)* @_except_handler3 {
entry:
  invoke void @may_throw_or_crash()
      to label %cont unwind label %lpad
cont:
  ret void
lpad:
  %cs = catchswitch within none [label %catch] unwind to caller
catch:
  %p = catchpad within %cs [i8* bitcast (i32 ()* @catchall_filt to i8*)]
  catchret from %p to label %cont
}

; CHECK-LABEL: _use_except_handler3:
; CHECK: pushl %ebp
; CHECK: movl %esp, %ebp
; CHECK: pushl %ebx
; CHECK: pushl %edi
; CHECK: pushl %esi
; CHECK: subl ${{[0-9]+}}, %esp
; CHECK: movl $-1, -16(%ebp)
; CHECK: movl $L__ehtable$use_except_handler3, -20(%ebp)
; CHECK: leal -28(%ebp), %[[node:[^ ,]*]]
; CHECK: movl $__except_handler3, -24(%ebp)
; CHECK: movl %fs:0, %[[next:[^ ,]*]]
; CHECK: movl %[[next]], -28(%ebp)
; CHECK: movl %[[node]], %fs:0
; CHECK: calll _may_throw_or_crash
; CHECK: movl -28(%ebp), %[[next:[^ ,]*]]
; CHECK: movl %[[next]], %fs:0
; CHECK: retl
; CHECK: LBB1_2: # %catch{{$}}

; CHECK: .section .xdata,"dr"
; CHECK-LABEL: L__ehtable$use_except_handler3:
; CHECK-NEXT:  .long   -1
; CHECK-NEXT:  .long   _catchall_filt
; CHECK-NEXT:  .long   LBB1_2

define void @use_except_handler4() personality i32 (...)* @_except_handler4 {
entry:
  invoke void @may_throw_or_crash()
      to label %cont unwind label %lpad
cont:
  ret void
lpad:
  %cs = catchswitch within none [label %catch] unwind to caller
catch:
  %p = catchpad within %cs [i8* bitcast (i32 ()* @catchall_filt to i8*)]
  catchret from %p to label %cont
}

; CHECK-LABEL: _use_except_handler4:
; CHECK: pushl %ebp
; CHECK: movl %esp, %ebp
; CHECK: subl ${{[0-9]+}}, %esp
; CHECK: movl %esp, -36(%ebp)
; CHECK: movl $-2, -16(%ebp)
; CHECK: movl $L__ehtable$use_except_handler4, %[[lsda:[^ ,]*]]
; CHECK: xorl ___security_cookie, %[[lsda]]
; CHECK: movl %[[lsda]], -20(%ebp)
; CHECK: leal -28(%ebp), %[[node:[^ ,]*]]
; CHECK: movl $__except_handler4, -24(%ebp)
; CHECK: movl %fs:0, %[[next:[^ ,]*]]
; CHECK: movl %[[next]], -28(%ebp)
; CHECK: movl %[[node]], %fs:0
; CHECK: calll _may_throw_or_crash
; CHECK: movl -28(%ebp), %[[next:[^ ,]*]]
; CHECK: movl %[[next]], %fs:0
; CHECK: retl
; CHECK: LBB2_2: # %catch{{$}}

; CHECK: .section .xdata,"dr"
; CHECK-LABEL: L__ehtable$use_except_handler4:
; CHECK-NEXT:  .long   -2
; CHECK-NEXT:  .long   0
; CHECK-NEXT:  .long   9999
; CHECK-NEXT:  .long   0
; CHECK-NEXT:  .long   -2
; CHECK-NEXT:  .long   _catchall_filt
; CHECK-NEXT:  .long   LBB2_2

define void @use_CxxFrameHandler3() personality i32 (...)* @__CxxFrameHandler3 {
  invoke void @may_throw_or_crash()
      to label %cont unwind label %catchall
cont:
  ret void

catchall:
  %cs = catchswitch within none [label %catch] unwind to caller
catch:
  %p = catchpad within %cs [i8* null, i32 64, i8* null]
  catchret from %p to label %cont
}

; CHECK-LABEL: _use_CxxFrameHandler3:
; CHECK: pushl %ebp
; CHECK: movl %esp, %ebp
; CHECK: subl ${{[0-9]+}}, %esp
; CHECK: movl %esp, -28(%ebp)
; CHECK: movl $-1, -16(%ebp)
; CHECK: leal -24(%ebp), %[[node:[^ ,]*]]
; CHECK: movl $___ehhandler$use_CxxFrameHandler3, -20(%ebp)
; CHECK: movl %fs:0, %[[next:[^ ,]*]]
; CHECK: movl %[[next]], -24(%ebp)
; CHECK: movl %[[node]], %fs:0
; CHECK: movl $0, -16(%ebp)
; CHECK: calll _may_throw_or_crash
; CHECK: movl -24(%ebp), %[[next:[^ ,]*]]
; CHECK: movl %[[next]], %fs:0
; CHECK: retl

; CHECK: .section .xdata,"dr"
; CHECK: .p2align 2
; CHECK-LABEL: L__ehtable$use_CxxFrameHandler3:
; CHECK-NEXT:  .long   429065506
; CHECK-NEXT:  .long   2
; CHECK-NEXT:  .long   ($stateUnwindMap$use_CxxFrameHandler3)
; CHECK-NEXT:  .long   1
; CHECK-NEXT:  .long   ($tryMap$use_CxxFrameHandler3)
; CHECK-NEXT:  .long   0
; CHECK-NEXT:  .long   0
; CHECK-NEXT:  .long   0
; CHECK-NEXT:  .long   1

; CHECK-LABEL: ___ehhandler$use_CxxFrameHandler3:
; CHECK: movl $L__ehtable$use_CxxFrameHandler3, %eax
; CHECK: jmp  ___CxxFrameHandler3 # TAILCALL

; CHECK: .safeseh __except_handler3
; CHECK: .safeseh __except_handler4
; CHECK: .safeseh ___ehhandler$use_CxxFrameHandler3
