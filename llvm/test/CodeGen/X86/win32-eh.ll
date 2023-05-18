; RUN: llc -mtriple=i686-pc-windows-msvc < %s | FileCheck %s

declare void @may_throw_or_crash()
declare i32 @_except_handler3(...)
declare i32 @_except_handler4(...)
declare i32 @__CxxFrameHandler3(...)
declare i32 @llvm.eh.typeid.for(ptr)

define internal i32 @catchall_filt() {
  ret i32 1
}

define void @use_except_handler3() personality ptr @_except_handler3 {
entry:
  invoke void @may_throw_or_crash()
      to label %cont unwind label %lpad
cont:
  ret void
lpad:
  %cs = catchswitch within none [label %catch] unwind to caller
catch:
  %p = catchpad within %cs [ptr @catchall_filt]
  catchret from %p to label %cont
}

; CHECK-LABEL: _use_except_handler3:
; CHECK: pushl %ebp
; CHECK-NEXT: movl %esp, %ebp
; CHECK-NEXT: pushl %ebx
; CHECK-NEXT: pushl %edi
; CHECK-NEXT: pushl %esi
; CHECK-NEXT: subl ${{[0-9]+}}, %esp
; CHECK-NEXT: movl %esp, -36(%ebp)
; CHECK-NEXT: movl $-1, -16(%ebp)
; CHECK-NEXT: movl $L__ehtable$use_except_handler3, -20(%ebp)
; CHECK-NEXT: leal -28(%ebp), %[[node:[^ ,]*]]
; CHECK-NEXT: movl $__except_handler3, -24(%ebp)
; CHECK-NEXT: movl %fs:0, %[[next:[^ ,]*]]
; CHECK-NEXT: movl %[[next]], -28(%ebp)
; CHECK-NEXT: movl %[[node]], %fs:0
; CHECK-NEXT: movl $0, -16(%ebp)
; CHECK-NEXT: calll _may_throw_or_crash

; CHECK: movl -28(%ebp), %[[next:[^ ,]*]]
; CHECK-NEXT: movl %[[next]], %fs:0
; CHECK: retl
; CHECK-NEXT: LBB1_2: # %catch{{$}}

; CHECK: .section .xdata,"dr"
; CHECK-LABEL: L__ehtable$use_except_handler3:
; CHECK-NEXT:  .long   -1
; CHECK-NEXT:  .long   _catchall_filt
; CHECK-NEXT:  .long   LBB1_2

define void @use_except_handler4() personality ptr @_except_handler4 {
entry:
  invoke void @may_throw_or_crash()
      to label %cont unwind label %lpad
cont:
  ret void
lpad:
  %cs = catchswitch within none [label %catch] unwind to caller
catch:
  %p = catchpad within %cs [ptr @catchall_filt]
  catchret from %p to label %cont
}

; CHECK-LABEL: _use_except_handler4:
; CHECK: pushl %ebp
; CHECK-NEXT: movl %esp, %ebp
; CHECK-NEXT: pushl %ebx
; CHECK-NEXT: pushl %edi
; CHECK-NEXT: pushl %esi
; CHECK-NEXT: subl ${{[0-9]+}}, %esp
; CHECK-NEXT: movl %ebp, %eax
; CHECK-NEXT: movl %esp, -36(%ebp)
; CHECK-NEXT: movl $-2, -16(%ebp)
; CHECK-NEXT: movl $L__ehtable$use_except_handler4, %[[lsda:[^ ,]*]]
; CHECK-NEXT: movl ___security_cookie, %[[seccookie:[^ ,]*]]
; CHECK-NEXT: xorl %[[seccookie]], %[[lsda]]
; CHECK-NEXT: movl %[[lsda]], -20(%ebp)
; CHECK-NEXT: xorl %[[seccookie]], %[[tmp1:[^ ,]*]]
; CHECK-NEXT: movl %[[tmp1]], -40(%ebp)
; CHECK-NEXT: leal -28(%ebp), %[[node:[^ ,]*]]
; CHECK-NEXT: movl $__except_handler4, -24(%ebp)
; CHECK-NEXT: movl %fs:0, %[[next:[^ ,]*]]
; CHECK-NEXT: movl %[[next]], -28(%ebp)
; CHECK-NEXT: movl %[[node]], %fs:0
; CHECK-NEXT: movl $0, -16(%ebp)
; CHECK-NEXT: calll _may_throw_or_crash

; CHECK: movl -28(%ebp), %[[next:[^ ,]*]]
; CHECK-NEXT: movl %[[next]], %fs:0
; CHECK-NEXT: addl $28, %esp
; CHECK-NEXT: popl %esi
; CHECK-NEXT: popl %edi
; CHECK-NEXT: popl %ebx
; CHECK-NEXT: popl %ebp
; CHECK-NEXT: retl
; CHECK-NEXT: LBB2_2: # %catch{{$}}

; CHECK: .section .xdata,"dr"
; CHECK-LABEL: L__ehtable$use_except_handler4:
; CHECK-NEXT:  .long   -2
; CHECK-NEXT:  .long   0
; CHECK-NEXT:  .long   -40
; CHECK-NEXT:  .long   0
; CHECK-NEXT:  .long   -2
; CHECK-NEXT:  .long   _catchall_filt
; CHECK-NEXT:  .long   LBB2_2

define void @use_except_handler4_ssp() sspstrong personality ptr @_except_handler4 {
entry:
  invoke void @may_throw_or_crash()
      to label %cont unwind label %lpad
cont:
  ret void
lpad:
  %cs = catchswitch within none [label %catch] unwind to caller
catch:
  %p = catchpad within %cs [ptr @catchall_filt]
  catchret from %p to label %cont
}

; CHECK-LABEL: _use_except_handler4_ssp:
; CHECK: pushl %ebp
; CHECK-NEXT: movl %esp, %ebp
; CHECK-NEXT: pushl %ebx
; CHECK-NEXT: pushl %edi
; CHECK-NEXT: pushl %esi
; CHECK-NEXT: subl ${{[0-9]+}}, %esp
; CHECK-NEXT: movl %ebp, %[[ehguard:[^ ,]*]]
; CHECK-NEXT: movl %esp, -36(%ebp)
; CHECK-NEXT: movl $-2, -16(%ebp)
; CHECK-NEXT: movl $L__ehtable$use_except_handler4_ssp, %[[lsda:[^ ,]*]]
; CHECK-NEXT: movl ___security_cookie, %[[seccookie:[^ ,]*]]
; CHECK-NEXT: xorl %[[seccookie]], %[[lsda]]
; CHECK-NEXT: movl %[[lsda]], -20(%ebp)
; CHECK-NEXT: xorl %[[seccookie]], %[[ehguard]]
; CHECK-NEXT: movl %[[ehguard]], -40(%ebp)
; CHECK-NEXT: leal -28(%ebp), %[[node:[^ ,]*]]
; CHECK-NEXT: movl $__except_handler4, -24(%ebp)
; CHECK-NEXT: movl %fs:0, %[[next:[^ ,]*]]
; CHECK-NEXT: movl %[[next]], -28(%ebp)
; CHECK-NEXT: movl %[[node]], %fs:0
; CHECK-NEXT: movl $0, -16(%ebp)
; CHECK-NEXT: calll _may_throw_or_crash
; CHECK: movl -28(%ebp), %[[next:[^ ,]*]]
; CHECK-NEXT: movl %[[next]], %fs:0   
; CHECK: retl
; CHECK-NEXT: [[catch:[^ ,]*]]: # %catch{{$}}



; CHECK: .section .xdata,"dr"
; CHECK-LABEL: L__ehtable$use_except_handler4_ssp:
; CHECK-NEXT:  .long   -2
; CHECK-NEXT:  .long   0
; CHECK-NEXT:  .long   -40  
; CHECK-NEXT:  .long   0
; CHECK-NEXT:  .long   -2
; CHECK-NEXT:  .long   _catchall_filt
; CHECK-NEXT:  .long   [[catch]]

define void @use_CxxFrameHandler3() personality ptr @__CxxFrameHandler3 {
  invoke void @may_throw_or_crash()
      to label %cont unwind label %catchall
cont:
  ret void

catchall:
  %cs = catchswitch within none [label %catch] unwind to caller
catch:
  %p = catchpad within %cs [ptr null, i32 64, ptr null]
  catchret from %p to label %cont
}

; CHECK-LABEL: _use_CxxFrameHandler3:
; CHECK: pushl %ebp
; CHECK-NEXT: movl %esp, %ebp
; CHECK-NEXT: pushl %ebx
; CHECK-NEXT: pushl %edi
; CHECK-NEXT: pushl %esi
; CHECK-NEXT: subl ${{[0-9]+}}, %esp
; CHECK-NEXT: movl %esp, -28(%ebp)
; CHECK-NEXT: movl $-1, -16(%ebp)
; CHECK-NEXT: leal -24(%ebp), %[[node:[^ ,]*]]
; CHECK-NEXT: movl $___ehhandler$use_CxxFrameHandler3, -20(%ebp)
; CHECK-NEXT: movl %fs:0, %[[next:[^ ,]*]]
; CHECK-NEXT: movl %[[next]], -24(%ebp)
; CHECK-NEXT: movl %[[node]], %fs:0
; CHECK-NEXT: movl $0, -16(%ebp)
; CHECK-NEXT: calll _may_throw_or_crash
; CHECK: movl -24(%ebp), %[[next:[^ ,]*]]
; CHECK-NEXT: movl %[[next]], %fs:0
; CHECK: retl

; CHECK: .section .xdata,"dr"
; CHECK-NEXT: .p2align 2
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
; CHECK-NEXT: jmp  ___CxxFrameHandler3 # TAILCALL

; CHECK: .safeseh __except_handler3
; CHECK-NEXT: .safeseh __except_handler4
; CHECK-NEXT: .safeseh ___ehhandler$use_CxxFrameHandler3
