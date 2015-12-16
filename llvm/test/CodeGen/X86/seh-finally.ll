; RUN: llc -mtriple=x86_64-windows-msvc < %s | FileCheck %s --check-prefix=X64
; RUN: sed -e 's/__C_specific_handler/_except_handler3/' %s | \
; RUN:        llc -mtriple=i686-windows-msvc | FileCheck %s --check-prefix=X86

@str_recovered = internal unnamed_addr constant [10 x i8] c"recovered\00", align 1

declare void @crash()

define i32 @main() personality i8* bitcast (i32 (...)* @__C_specific_handler to i8*) {
entry:
  invoke void @crash()
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %entry
  %call = call i32 @puts(i8* getelementptr inbounds ([10 x i8], [10 x i8]* @str_recovered, i64 0, i64 0))
  call void @abort()
  ret i32 0

lpad:                                             ; preds = %entry
  %p = cleanuppad within none []
  %call2 = call i32 @puts(i8* getelementptr inbounds ([10 x i8], [10 x i8]* @str_recovered, i64 0, i64 0)) [ "funclet"(token %p) ]
  cleanupret from %p unwind to caller
}

; X64-LABEL: main:
; X64: retq

; X64: .seh_handlerdata
; X64-NEXT: .Lmain$parent_frame_offset = 32
; X64-NEXT: .long   (.Llsda_end0-.Llsda_begin0)/16
; X64-NEXT: .Llsda_begin0:
; X64-NEXT: .long   .Ltmp0@IMGREL+1
; X64-NEXT: .long   .Ltmp1@IMGREL+1
; X64-NEXT: .long   "?dtor$2@?0?main@4HA"@IMGREL
; X64-NEXT: .long   0
; X64-NEXT: .Llsda_end0:

; X64-LABEL: "?dtor$2@?0?main@4HA":
; X64: callq puts
; X64: retq

; X86-LABEL: _main:
; X86: retl

; X86-LABEL: "?dtor$2@?0?main@4HA":
; X86: LBB0_2:
; X86: calll _puts
; X86: retl

; X86: .section .xdata,"dr"
; X86: L__ehtable$main:
; X86-NEXT: .long -1
; X86-NEXT: .long 0
; X86-NEXT: .long LBB0_2

declare i32 @__C_specific_handler(...)

declare i32 @puts(i8*)

declare void @abort()
