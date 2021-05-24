; RUN: llc -O0 -mtriple=x86_64-apple-darwin12 -filetype=obj -o - %s | llvm-readobj -S - | FileCheck %s
; Test that we emit weak_odr thread_locals correctly into the thread_bss section
; PR15972

; CHECK: Section {
; CHECK:   Index: 1
; CHECK:   Name: __thread_bss (5F 5F 74 68 72 65 61 64 5F 62 73 73 00 00 00 00)
; CHECK:   Size: 0x8
; CHECK:   Alignment: 3
; CHECK: }
; CHECK: Section {
; CHECK:   Index: 2
; CHECK:   Name: __thread_vars (5F 5F 74 68 72 65 61 64 5F 76 61 72 73 00 00 00)

; Generated from this C++ source
; template<class T>
; struct Tls {
;   static __thread void* val;
; };

; template<class T> __thread void* Tls<T>::val;

; void* f(int x) {
;         return Tls<long>::val;
; }

@_ZN3TlsIlE3valE = weak_odr thread_local global i8* null, align 8

; Function Attrs: nounwind ssp uwtable
define i8* @_Z1fi(i32 %x) #0 {
entry:
  %x.addr = alloca i32, align 4
  store i32 %x, i32* %x.addr, align 4
  %0 = load i8*, i8** @_ZN3TlsIlE3valE, align 8
  ret i8* %0
}

attributes #0 = { nounwind ssp uwtable "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
