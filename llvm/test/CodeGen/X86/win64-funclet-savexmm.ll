; RUN: llc -mtriple=x86_64-pc-windows-msvc -mattr=+avx < %s | FileCheck %s

; void bar(int a, int b, int c, int d, int e);
; void baz(int x);
; 
; void foo(int a, int b, int c, int d, int e)
; {
;   __asm("nop" ::: "bx", "cx", "xmm5", "xmm6", "ymm7");
;   try {
;     bar(a, b, c, d, e);
;   }
;   catch (...) {
;     baz(a);
;     if (a)
;       __asm("nop" ::: "xmm8");
;   }
; }

%rtti.TypeDescriptor2 = type { ptr, ptr, [3 x i8] }

$"??_R0H@8" = comdat any

@"??_7type_info@@6B@" = external constant ptr
@"??_R0H@8" = linkonce_odr global %rtti.TypeDescriptor2 { ptr @"??_7type_info@@6B@", ptr null, [3 x i8] c".H\00" }, comdat

declare dso_local i32 @__CxxFrameHandler3(...)
declare dso_local void @"?bar@@YAXHHHHH@Z"(i32, i32, i32, i32, i32)
declare dso_local void @"?baz@@YAXH@Z"(i32)

define dso_local void @"?foo@@YAXHHHHH@Z"(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e) personality ptr @__CxxFrameHandler3 {
entry:
  %e.addr = alloca i32, align 4
  %d.addr = alloca i32, align 4
  %c.addr = alloca i32, align 4
  %b.addr = alloca i32, align 4
  %a.addr = alloca i32, align 4
  store i32 %e, ptr %e.addr, align 4
  store i32 %d, ptr %d.addr, align 4
  store i32 %c, ptr %c.addr, align 4
  store i32 %b, ptr %b.addr, align 4
  store i32 %a, ptr %a.addr, align 4
  call void asm sideeffect "nop", "~{bx},~{cx},~{xmm5},~{xmm6},~{ymm7}"()
  %0 = load i32, ptr %e.addr, align 4
  %1 = load i32, ptr %d.addr, align 4
  %2 = load i32, ptr %c.addr, align 4
  %3 = load i32, ptr %b.addr, align 4
  %4 = load i32, ptr %a.addr, align 4
  invoke void @"?bar@@YAXHHHHH@Z"(i32 %4, i32 %3, i32 %2, i32 %1, i32 %0)
          to label %invoke.cont unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %5 = catchswitch within none [label %catch] unwind to caller

catch:                                            ; preds = %catch.dispatch
  %6 = catchpad within %5 [ptr null, i32 64, ptr null]
  %7 = load i32, ptr %a.addr, align 4
  call void @"?baz@@YAXH@Z"(i32 %7) [ "funclet"(token %6) ]
  %8 = load i32, ptr %a.addr, align 4
  %tobool = icmp ne i32 %8, 0
  br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %catch
  call void asm sideeffect "nop", "~{xmm8}"() [ "funclet"(token %6) ]
  br label %if.end

invoke.cont:                                      ; preds = %entry
  br label %try.cont

if.end:                                           ; preds = %if.then, %catch
  catchret from %6 to label %catchret.dest

catchret.dest:                                    ; preds = %if.end
  br label %try.cont

try.cont:                                         ; preds = %catchret.dest, %invoke.cont
  ret void
}

; CHECK: # %catch
; CHECK: movq    %rdx, 16(%rsp)
; CHECK: pushq   %rbp
; CHECK: .seh_pushreg %rbp
; CHECK: pushq   %rbx
; CHECK: .seh_pushreg %rbx
; CHECK: subq    $88, %rsp
; CHECK: .seh_stackalloc 88
; CHECK: leaq    112(%rdx), %rbp
; CHECK: vmovaps %xmm8, 32(%rsp)
; CHECK: .seh_savexmm %xmm8, 32
; CHECK: vmovaps %xmm7, 48(%rsp)
; CHECK: .seh_savexmm %xmm7, 48
; CHECK: vmovaps %xmm6, 64(%rsp)
; CHECK: .seh_savexmm %xmm6, 64
; CHECK: .seh_endprologue
; CHECK: movl   -{{[0-9]+}}(%rbp), %ecx
; CHECK: vmovaps 64(%rsp), %xmm6
; CHECK: vmovaps 48(%rsp), %xmm7
; CHECK: vmovaps 32(%rsp), %xmm8
; CHECK: leaq    .LBB0_1(%rip), %rax
; CHECK: addq    $88, %rsp
; CHECK: popq    %rbx
; CHECK: popq    %rbp
; CHECK: retq # CATCHRET

; CHECK-LABEL: "$handlerMap$0$?foo@@YAXHHHHH@Z":
; CHECK-NEXT: .long   64                      # Adjectives
; CHECK-NEXT: .long   0                       # Type
; CHECK-NEXT: .long   0                       # CatchObjOffset
; CHECK-NEXT: .long   "?catch$2@?0??foo@@YAXHHHHH@Z@4HA"@IMGREL # Handler
; Sum of:
;   16 RDX store offset
;   16 two pushes
;   72 stack alloc
; CHECK-NEXT: .long   120                     # ParentFrameOffset

