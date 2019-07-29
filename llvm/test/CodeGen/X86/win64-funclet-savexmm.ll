; RUN: llc -mtriple=x86_64-pc-windows-msvc -mattr=+avx < %s | FileCheck %s

; void foo(void)
; {
;   __asm("nop" ::: "bx", "cx", "xmm5", "xmm6", "ymm7");
;   try {
;     throw;
;   }
;   catch (int x) {
;   }
; }

%rtti.TypeDescriptor2 = type { i8**, i8*, [3 x i8] }
%eh.ThrowInfo = type { i32, i8*, i8*, i8* }

$"??_R0H@8" = comdat any

@"??_7type_info@@6B@" = external constant i8*
@"??_R0H@8" = linkonce_odr global %rtti.TypeDescriptor2 { i8** @"??_7type_info@@6B@", i8* null, [3 x i8] c".H\00" }, comdat

declare dso_local i32 @__CxxFrameHandler3(...)
declare dso_local x86_stdcallcc void @_CxxThrowException(i8*, %eh.ThrowInfo*)

define dso_local void @"?foo@@YAXXZ"() personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
entry:
  %x = alloca i32, align 4
  call void asm sideeffect "nop", "~{bx},~{cx},~{xmm5},~{xmm6},~{ymm7}"()
  invoke void @_CxxThrowException(i8* null, %eh.ThrowInfo* null)
          to label %unreachable unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %0 = catchswitch within none [label %catch] unwind to caller

catch:                                            ; preds = %catch.dispatch
  %1 = catchpad within %0 [%rtti.TypeDescriptor2* @"??_R0H@8", i32 0, i32* %x]
  catchret from %1 to label %catchret.dest

catchret.dest:                                    ; preds = %catch
  br label %try.cont

try.cont:                                         ; preds = %catchret.dest
  ret void

unreachable:                                      ; preds = %entry
  unreachable
}

; CHECK: # %catch
; CHECK: movq    %rdx, 16(%rsp)
; CHECK: pushq   %rbp
; CHECK: .seh_pushreg 5
; CHECK: pushq   %rbx
; CHECK: .seh_pushreg 3
; CHECK: subq    $72, %rsp
; CHECK: .seh_stackalloc 72
; CHECK: leaq    80(%rdx), %rbp
; CHECK: vmovaps %xmm7, 48(%rsp)
; CHECK: .seh_savexmm 7, 48
; CHECK: vmovaps %xmm6, 32(%rsp)
; CHECK: .seh_savexmm 6, 32
; CHECK: .seh_endprologue
; CHECK: vmovaps 32(%rsp), %xmm6
; CHECK: vmovaps 48(%rsp), %xmm7
; CHECK: leaq    .LBB0_3(%rip), %rax
; CHECK: addq    $72, %rsp
; CHECK: popq    %rbx
; CHECK: popq    %rbp
; CHECK: retq # CATCHRET
