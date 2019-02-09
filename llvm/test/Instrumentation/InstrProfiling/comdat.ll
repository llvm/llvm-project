; RUN: opt < %s -mtriple=x86_64-unknown-linux -instrprof -S | FileCheck %s --check-prefixes=ELF
; RUN: opt < %s  -mtriple=x86_64-pc-win32-coff -instrprof -S | FileCheck %s --check-prefixes=COFF
; RUN: opt < %s -mtriple=x86_64-unknown-linux -passes=instrprof -S | FileCheck %s --check-prefixes=ELF
; RUN: opt < %s  -mtriple=x86_64-pc-win32-coff -passes=instrprof -S | FileCheck %s --check-prefixes=COFF

; There are two main cases where comdats are necessary:
; 1. standard inline functions (weak_odr / linkonce_odr)
; 2. available externally functions (C99 inline / extern template / dllimport)
; Check that we do the right thing for the two object formats with comdats, ELF
; and COFF.

declare void @llvm.instrprof.increment(i8*, i64, i32, i32)

$foo_inline = comdat any

@__profn_foo_inline = linkonce_odr hidden constant [10 x i8] c"foo_inline"

; ELF: @__profc_foo_inline = linkonce_odr hidden global{{.*}}, section "__llvm_prf_cnts", comdat($__profv_foo_inline), align 8
; ELF: @__profd_foo_inline = linkonce_odr hidden global{{.*}}, section "__llvm_prf_data", comdat($__profv_foo_inline), align 8
; COFF: @__profc_foo_inline = internal global{{.*}}, section ".lprfc$M", comdat($foo_inline), align 8
; COFF: @__profd_foo_inline = internal global{{.*}}, section ".lprfd$M", comdat($foo_inline), align 8
define weak_odr void @foo_inline() comdat {
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([10 x i8], [10 x i8]* @__profn_foo_inline, i32 0, i32 0), i64 0, i32 1, i32 0)
  ret void
}

$foo_extern = comdat any

@__profn_foo_extern = linkonce_odr hidden constant [10 x i8] c"foo_extern"

; ELF: @__profc_foo_extern = linkonce_odr hidden global{{.*}}, section "__llvm_prf_cnts", comdat($__profv_foo_extern)
; ELF: @__profd_foo_extern = linkonce_odr hidden global{{.*}}, section "__llvm_prf_data", comdat($__profv_foo_extern)
; COFF: @__profc_foo_extern = linkonce_odr dso_local global{{.*}}, section ".lprfc$M", comdat, align 8
; COFF: @__profd_foo_extern = internal global{{.*}}, section ".lprfd$M", comdat($__profc_foo_extern), align 8
define available_externally void @foo_extern() {
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([10 x i8], [10 x i8]* @__profn_foo_extern, i32 0, i32 0), i64 0, i32 1, i32 0)
  ret void
}
