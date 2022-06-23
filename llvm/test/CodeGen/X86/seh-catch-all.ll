; RUN: llc -mtriple=x86_64-windows-msvc < %s | FileCheck %s

@str = linkonce_odr unnamed_addr constant [27 x i8] c"GetExceptionCode(): 0x%lx\0A\00", align 1

declare i32 @llvm.eh.exceptioncode(token)
declare i32 @__C_specific_handler(...)
declare void @crash()
declare i32 @printf(ptr nocapture readonly, ...) nounwind

define i32 @main() personality ptr @__C_specific_handler {
entry:
  invoke void @crash()
          to label %__try.cont unwind label %lpad

__try.cont:
  ret i32 0

lpad:
  %cs1 = catchswitch within none [label %catchall] unwind to caller

catchall:
  %p = catchpad within %cs1 [ptr null, i32 64, ptr null]
  %code = call i32 @llvm.eh.exceptioncode(token %p)
  call i32 (ptr, ...) @printf(ptr @str, i32 %code) [ "funclet"(token %p) ]
  catchret from %p to label %__try.cont
}

; Check that we can get the exception code from eax to the printf.

; CHECK-LABEL: main:
; CHECK: callq crash
; CHECK: retq
; CHECK: .LBB0_2: # %catchall
; CHECK: leaq str(%rip), %rcx
; CHECK: movl %eax, %edx
; CHECK: callq printf

; CHECK: .seh_handlerdata
; CHECK-NEXT: .Lmain$parent_frame_offset
; CHECK-NEXT: .long (.Llsda_end0-.Llsda_begin0)/16
; CHECK-NEXT: .Llsda_begin0:
; CHECK-NEXT: .long .Ltmp{{[0-9]+}}@IMGREL
; CHECK-NEXT: .long .Ltmp{{[0-9]+}}@IMGREL+1
; CHECK-NEXT: .long 1
; CHECK-NEXT: .long .LBB0_2@IMGREL
; CHECK-NEXT: .Llsda_end0:
