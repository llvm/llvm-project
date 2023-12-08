; Test that interposable symbols do not get put in comdats.
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=3 -sanitizer-coverage-trace-pc-guard -mtriple x86_64-linux-gnu -S | FileCheck %s --check-prefixes=CHECK,ELF
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=3 -sanitizer-coverage-trace-pc-guard -mtriple x86_64-windows-msvc -S | FileCheck %s --check-prefixes=CHECK,COFF

define void @Vanilla() {
entry:
  ret void
}

define linkonce void @LinkOnce() {
entry:
  ret void
}

define weak void @Weak() {
entry:
  ret void
}

declare extern_weak void @ExternWeak()

define linkonce_odr void @LinkOnceOdr() {
entry:
  ret void
}

define weak_odr void @WeakOdr() {
entry:
  ret void
}

; CHECK:      $Vanilla = comdat nodeduplicate
; ELF:        $LinkOnceOdr = comdat nodeduplicate
; COFF:       $LinkOnceOdr = comdat any
; CHECK:      @__sancov_gen_ = private global [1 x i32] zeroinitializer, section {{.*}}, comdat($Vanilla), align 4{{$}}
; CHECK-NEXT: @__sancov_gen_.1 = private global [1 x i32] zeroinitializer, section {{.*}}, align 4{{$}}
; CHECK-NEXT: @__sancov_gen_.2 = private global [1 x i32] zeroinitializer, section {{.*}}, align 4{{$}}
; CHECK-NEXT: @__sancov_gen_.3 = private global [1 x i32] zeroinitializer, section {{.*}}, comdat($LinkOnceOdr), align 4{{$}}
; CHECK-NEXT: @__sancov_gen_.4 = private global [1 x i32] zeroinitializer, section {{.*}}, comdat($WeakOdr), align 4{{$}}

; CHECK: define void @Vanilla() comdat {
; ELF:   define linkonce void @LinkOnce() comdat {
; ELF:   define weak void @Weak() comdat {
; COFF:  define linkonce void @LinkOnce() {
; COFF:  define weak void @Weak() {
; CHECK: declare extern_weak void @ExternWeak()
; CHECK: define linkonce_odr void @LinkOnceOdr() comdat {
; CHECK: define weak_odr void @WeakOdr() comdat {
