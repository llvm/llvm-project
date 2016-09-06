; RUN: llc -stack-symbol-ordering=0 < %s | FileCheck %s

; The aligned alloca means that we have to realign the stack, which forces the
; use of ESI to address local variables.

target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686--windows-msvc"

; Function Attrs: nounwind
define void @realigned_try() personality i8* bitcast (i32 (...)* @_except_handler3 to i8*) {
entry:
  %x = alloca [4 x i32], align 16
  %arrayidx = getelementptr inbounds [4 x i32], [4 x i32]* %x, i32 0, i32 0
  invoke void @useit(i32* %arrayidx)
          to label %__try.cont unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %cs1 = catchswitch within none [label %__except.ret] unwind to caller

__except.ret:                                     ; preds = %catch.dispatch
  %pad = catchpad within %cs1 [i8* bitcast (i32 ()* @"\01?filt$0@0@realigned_try@@" to i8*)]
  catchret from %pad to label %__try.cont

__try.cont:                                       ; preds = %entry, %__except.ret
  ret void
}

; Function Attrs: nounwind argmemonly

; Function Attrs: nounwind
define internal i32 @"\01?filt$0@0@realigned_try@@"() {
entry:
  ret i32 1
}

declare void @useit(i32*)

declare i32 @_except_handler3(...)

; CHECK-LABEL: _realigned_try:
; Prologue
; CHECK: pushl   %ebp
; CHECK: movl    %esp, %ebp
; CHECK: pushl   %ebx
; CHECK: pushl   %edi
; CHECK: pushl   %esi
; CHECK: andl    $-16, %esp
; CHECK: subl    $64, %esp
; CHECK: movl    %esp, %esi
; Spill EBP
; CHECK: movl    %ebp, 12(%esi)
; Spill ESP
; CHECK: movl    %esp, 36(%esi)
; The state is stored at ESI+56, the end of the node is ESI+60.
; CHECK: movl    $-1, 56(%esi)
;
; __try
; CHECK: calll _useit
;
; Epilogue
; CHECK: LBB0_2:       # %__try.cont
; CHECK: leal    -12(%ebp), %esp
; CHECK: popl    %esi
; CHECK: popl    %edi
; CHECK: popl    %ebx
; CHECK: popl    %ebp
; CHECK: retl
;
; CHECK: LBB0_1:                                 # %__except.ret
; Restore ESP
; CHECK: movl    -24(%ebp), %esp
; Recompute ESI by subtracting 60 from the end of the registration node.
; CHECK: leal    -60(%ebp), %esi
; Restore EBP
; CHECK: movl    12(%esi), %ebp
; Rejoin normal control flow
; CHECK: jmp     LBB0_2
