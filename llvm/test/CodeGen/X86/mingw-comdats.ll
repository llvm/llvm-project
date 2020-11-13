; RUN: llc -function-sections -mtriple=x86_64-windows-itanium < %s | FileCheck %s
; RUN: llc -function-sections -mtriple=x86_64-windows-msvc < %s | FileCheck %s
; RUN: llc -function-sections -mtriple=x86_64-w64-windows-gnu < %s | FileCheck %s --check-prefix=GNU
; RUN: llc -function-sections -mtriple=i686-w64-windows-gnu < %s | FileCheck %s --check-prefix=GNU32
; RUN: llc -function-sections -mtriple=x86_64-w64-windows-gnu < %s -filetype=obj | llvm-objdump - --headers | FileCheck %s --check-prefix=GNUOBJ

; GCC and MSVC handle comdats completely differently. Make sure we do the right
; thing for each.

; Modeled on this C++ source, with additional modifications for
; -ffunction-sections:
; int bar(int);
; __declspec(selectany) int gv = 42;
; inline int foo(int x) { return bar(x) + gv; }
; int main() { return foo(1); }

$_Z3fooi = comdat any

$gv = comdat any

@gv = weak_odr dso_local global i32 42, comdat, align 4

; Function Attrs: norecurse uwtable
define dso_local i32 @main() #0 {
entry:
  %call = tail call i32 @_Z3fooi(i32 1)
  ret i32 %call
}

; CHECK: .section        .text,"xr",one_only,main
; CHECK: main:
; GNU: .section        .text$main,"xr",one_only,main
; GNU: main:
; GNU32: .section        .text$main,"xr",one_only,_main
; GNU32: _main:

define dso_local x86_fastcallcc i32 @fastcall(i32 %x, i32 %y) {
  %rv = add i32 %x, %y
  ret i32 %rv
}

; CHECK: .section        .text,"xr",one_only,fastcall
; CHECK: fastcall:
; GNU: .section        .text$fastcall,"xr",one_only,fastcall
; GNU: fastcall:
; GNU32: .section        .text$fastcall,"xr",one_only,@fastcall@8
; GNU32: @fastcall@8:

; Function Attrs: inlinehint uwtable
define linkonce_odr dso_local i32 @_Z3fooi(i32 %x) #1 comdat {
entry:
  %call = tail call i32 @_Z3bari(i32 %x)
  %0 = load i32, i32* @gv, align 4
  %add = add nsw i32 %0, %call
  ret i32 %add
}

; CHECK: .section        .text,"xr",discard,_Z3fooi
; CHECK: _Z3fooi:
; CHECK: .section        .data,"dw",discard,gv
; CHECK: gv:
; CHECK: .long 42

; GNU: .section        .text$_Z3fooi,"xr",discard,_Z3fooi
; GNU: _Z3fooi:
; GNU: .section        .data$gv,"dw",discard,gv
; GNU: gv:
; GNU: .long 42

; GNU32: .section        .text$_Z3fooi,"xr",discard,__Z3fooi
; GNU32: __Z3fooi:
; GNU32: .section        .data$gv,"dw",discard,_gv
; GNU32: _gv:
; GNU32: .long 42


; Make sure the assembler puts the .xdata and .pdata in sections with the right
; names.
; GNUOBJ: .text$_Z3fooi
; GNUOBJ: .xdata$_Z3fooi
; GNUOBJ: .data$gv
; GNUOBJ: .pdata$_Z3fooi

declare dso_local i32 @_Z3bari(i32)

attributes #0 = { norecurse uwtable }
attributes #1 = { inlinehint uwtable }
