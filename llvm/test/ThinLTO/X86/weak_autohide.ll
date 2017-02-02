; RUN: opt -module-summary %s -o %t1.bc
; RUN: opt -module-summary %p/Inputs/weak_autohide.ll -o %t2.bc

; RUN: llvm-lto2 %t1.bc %t2.bc -o %t.o -save-temps \
; RUN:     -r=%t1.bc,_strong_func,pxl \
; RUN:     -r=%t1.bc,_weakodrfunc,pl \
; RUN:     -r=%t2.bc,_weakodrfunc,l
; RUN: llvm-dis < %t.o.0.2.internalize.bc | FileCheck  %s --check-prefix=AUTOHIDE


; AUTOHIDE: weak_odr hidden void @weakodrfunc

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

define weak_odr void @weakodrfunc() #0 {
  ret void
}

define void @strong_func() #0 {
	call void @weakodrfunc()
	ret void
}

