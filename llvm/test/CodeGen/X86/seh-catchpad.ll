; RUN: llc < %s | FileCheck %s

; Based on the source:
; extern "C" int puts(const char *);
; extern "C" int printf(const char *, ...);
; extern "C" int do_div(int a, int b) { return a / b; }
; extern "C" int filt();
; int main() {
;   __try {
;     __try {
;       do_div(1, 0);
;     } __except (1) {
;       __try {
;         do_div(1, 0);
;       } __finally {
;         puts("finally");
;       }
;     }
;   } __except (filt()) {
;     puts("caught");
;   }
;   return 0;
; }

; ModuleID = 't.cpp'
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

$"\01??_C@_07MKBLAIAL@finally?$AA@" = comdat any

$"\01??_C@_06IBDBCMGJ@caught?$AA@" = comdat any

@"\01??_C@_07MKBLAIAL@finally?$AA@" = linkonce_odr unnamed_addr constant [8 x i8] c"finally\00", comdat, align 1
@"\01??_C@_06IBDBCMGJ@caught?$AA@" = linkonce_odr unnamed_addr constant [7 x i8] c"caught\00", comdat, align 1

; Function Attrs: nounwind readnone
define i32 @do_div(i32 %a, i32 %b) #0 {
entry:
  %div = sdiv i32 %a, %b
  ret i32 %div
}

define i32 @main() #1 personality i8* bitcast (i32 (...)* @__C_specific_handler to i8*) {
entry:
  %call = invoke i32 @do_div(i32 1, i32 0) #4
          to label %__try.cont.12 unwind label %catch.dispatch

__except.2:                                       ; preds = %__except
  %call4 = invoke i32 @do_div(i32 1, i32 0) #4
          to label %invoke.cont.3 unwind label %ehcleanup

invoke.cont.3:                                    ; preds = %__except.2
  invoke fastcc void @"\01?fin$0@0@main@@"() #4
          to label %__try.cont.12 unwind label %catch.dispatch.7

__except.9:                                       ; preds = %__except.ret
  %call11 = tail call i32 @puts(i8* nonnull getelementptr inbounds ([7 x i8], [7 x i8]* @"\01??_C@_06IBDBCMGJ@caught?$AA@", i64 0, i64 0))
  br label %__try.cont.12

__try.cont.12:                                    ; preds = %invoke.cont.3, %entry, %__except.9
  ret i32 0

catch.dispatch:                                   ; preds = %entry
  %cs1 = catchswitch within none [label %__except] unwind label %catch.dispatch.7

__except:                                         ; preds = %catch.dispatch
  %cp1 = catchpad within %cs1 [i8* null]
  catchret from %cp1 to label %__except.2

ehcleanup:                                        ; preds = %__except.2
  %cp2 = cleanuppad within none []
  invoke fastcc void @"\01?fin$0@0@main@@"() #4 [ "funclet"(token %cp2) ]
          to label %invoke.cont.6 unwind label %catch.dispatch.7

invoke.cont.6:                                    ; preds = %ehcleanup
  cleanupret from %cp2 unwind label %catch.dispatch.7

catch.dispatch.7:
  %cs2 = catchswitch within none [label %__except.ret] unwind to caller

__except.ret:                                     ; preds = %catch.dispatch.7
  %cp3 = catchpad within %cs2 [i8* bitcast (i32 (i8*, i8*)* @"\01?filt$0@0@main@@" to i8*)]
  catchret from %cp3 to label %__except.9
}

; CHECK: main:                                   # @main
; CHECK: .seh_proc main
; CHECK:         .seh_handler __C_specific_handler, @unwind, @except
; CHECK:         pushq   %rbp
; CHECK:         .seh_pushreg %rbp
; CHECK:         subq    $32, %rsp
; CHECK:         .seh_stackalloc 32
; CHECK:         leaq    32(%rsp), %rbp
; CHECK:         .seh_setframe %rbp, 32
; CHECK:         .seh_endprologue
; CHECK: .Ltmp0:
; CHECK:         movl    $1, %ecx
; CHECK:         xorl    %edx, %edx
; CHECK:         callq   do_div
; CHECK: .Ltmp1:
; CHECK: .LBB1_[[epilogue:[0-9]+]]:                                # %__try.cont.12
; CHECK:         xorl    %eax, %eax
; CHECK:         addq    $32, %rsp
; CHECK:         popq    %rbp
; CHECK:         retq
; CHECK: .LBB1_[[except1bb:[0-9]+]]:                                # %__except
; CHECK: .Ltmp2:
; CHECK:         movl    $1, %ecx
; CHECK:         xorl    %edx, %edx
; CHECK:         callq   do_div
; CHECK: .Ltmp3:
; CHECK:         callq   "?fin$0@0@main@@"
; CHECK:         jmp     .LBB1_[[epilogue]]
; CHECK: .LBB1_[[except2bb:[0-9]+]]:                                # %__except.ret
; CHECK:         leaq    "??_C@_06IBDBCMGJ@caught?$AA@"(%rip), %rcx
; CHECK:         callq   puts
; CHECK:         jmp     .LBB1_[[epilogue]]

; CHECK:         .seh_handlerdata
; CHECK-NEXT:         .set .Lmain$parent_frame_offset, 32
; CHECK-NEXT:         .long   (.Llsda_end0-.Llsda_begin0)/16
; CHECK-NEXT: .Llsda_begin0:
; CHECK-NEXT:         .long   .Ltmp0@IMGREL+1
; CHECK-NEXT:         .long   .Ltmp1@IMGREL+1
; CHECK-NEXT:         .long   1
; CHECK-NEXT:         .long   .LBB1_[[except1bb]]@IMGREL
; CHECK-NEXT:         .long   .Ltmp0@IMGREL+1
; CHECK-NEXT:         .long   .Ltmp1@IMGREL+1
; CHECK-NEXT:         .long   "?filt$0@0@main@@"@IMGREL
; CHECK-NEXT:         .long   .LBB1_[[except2bb]]@IMGREL
; CHECK-NEXT:         .long   .Ltmp2@IMGREL+1
; CHECK-NEXT:         .long   .Ltmp3@IMGREL+1
; CHECK-NEXT:         .long   "?dtor$[[finbb:[0-9]+]]@?0?main@4HA"@IMGREL
; CHECK-NEXT:         .long   0
; CHECK-NEXT:         .long   .Ltmp2@IMGREL+1
; CHECK-NEXT:         .long   .Ltmp3@IMGREL+1
; CHECK-NEXT:         .long   "?filt$0@0@main@@"@IMGREL
; CHECK-NEXT:         .long   .LBB1_3@IMGREL
; CHECK-NEXT:         .long   .Ltmp6@IMGREL+1
; CHECK-NEXT:         .long   .Ltmp7@IMGREL+1
; CHECK-NEXT:         .long   "?filt$0@0@main@@"@IMGREL
; CHECK-NEXT:         .long   .LBB1_3@IMGREL
; CHECK-NEXT: .Llsda_end0:

; CHECK:         .text
; CHECK:         .seh_endproc

; CHECK: "?dtor$[[finbb]]@?0?main@4HA":
; CHECK: .seh_proc "?dtor$[[finbb]]@?0?main@4HA"
; CHECK-NOT:         .seh_handler
; CHECK: .LBB1_[[finbb]]:                                # %ehcleanup
; CHECK:         movq    %rdx, 16(%rsp)
; CHECK:         pushq   %rbp
; CHECK:         .seh_pushreg %rbp
; CHECK:         subq    $32, %rsp
; CHECK:         .seh_stackalloc 32
; CHECK:         leaq    32(%rdx), %rbp
; CHECK:         .seh_endprologue
; CHECK:         callq   "?fin$0@0@main@@"
; CHECK:         nop
; CHECK:         addq    $32, %rsp
; CHECK:         popq    %rbp
; CHECK:         retq
; CHECK:         .seh_handlerdata
; CHECK:         .seh_endproc

define internal i32 @"\01?filt$0@0@main@@"(i8* nocapture readnone %exception_pointers, i8* nocapture readnone %frame_pointer) #1 {
entry:
  %call = tail call i32 @filt()
  ret i32 %call
}

; CHECK: "?filt$0@0@main@@":                     # @"\01?filt$0@0@main@@"
; CHECK:         jmp       filt  # TAILCALL

declare i32 @filt() #1

declare i32 @__C_specific_handler(...)

; Function Attrs: noinline nounwind
define internal fastcc void @"\01?fin$0@0@main@@"() #2 {
entry:
  %call = tail call i32 @puts(i8* getelementptr inbounds ([8 x i8], [8 x i8]* @"\01??_C@_07MKBLAIAL@finally?$AA@", i64 0, i64 0)) #5
  ret void
}

; Function Attrs: nounwind
declare i32 @puts(i8* nocapture readonly) #3

attributes #0 = { nounwind readnone "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-features"="+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-features"="+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { noinline nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-features"="+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-features"="+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { noinline }
attributes #5 = { nounwind }
