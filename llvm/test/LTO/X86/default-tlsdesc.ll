;; Test that Fuchsia enables TLSDESC by default during LTO.
; RUN: opt -mtriple=x86_64-unknown-fuchsia < %s > %t.bc
; RUN: llvm-lto -exported-symbol=f -relocation-model=pic -o %t.o %t.bc
; RUN: llvm-readelf -r %t.o | FileCheck %s --check-prefix=FUCHSIA

; RUN: opt -mtriple=x86_64-unknown-fuchsia -module-summary -o %t.thin.bc %s
; RUN: llvm-lto2 run -r %t.thin.bc,f,plx -r %t.thin.bc,x, -relocation-model=pic -o %t.thin.o %t.thin.bc
; RUN: llvm-readelf -r %t.thin.o.1 | FileCheck %s --check-prefix=FUCHSIA

;; Test that Linux does not enable TLSDESC by default during LTO.
; RUN: llvm-lto2 run -r %t.thin.bc,f,plx -r %t.thin.bc,x, -override-triple=x86_64-unknown-linux -relocation-model=pic -o %t.linux.o %t.thin.bc
; RUN: llvm-readelf -r %t.linux.o.1 | FileCheck %s --check-prefix=LINUX

declare ptr @llvm.threadlocal.address.p0(ptr)

@x = external thread_local global i32, align 4

define ptr @f() {
entry:
  %1 = tail call ptr @llvm.threadlocal.address.p0(ptr @x)
  ; FUCHSIA: R_X86_64_GOTPC32_TLSDESC
  ; FUCHSIA: R_X86_64_TLSDESC_CALL
  ; LINUX: R_X86_64_TLSGD
  ; LINUX: R_X86_64_PLT32 {{.*}}__tls_get_addr
  ret ptr %1
}
