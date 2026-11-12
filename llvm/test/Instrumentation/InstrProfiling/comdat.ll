; There are two main cases where comdats are necessary:
; 1. standard inline functions (weak_odr / linkonce_odr)
; 2. available externally functions (C99 inline / extern template / dllimport)
; Check that we do the right thing for the two object formats with comdats, ELF
; and COFF.
;
; Test comdat functions. Non-comdat functions are tested in linkage.ll.
; RUN: split-file %s %t
; RUN: cat %t/main.ll %t/disable.ll > %t0.ll
; RUN: cat %t/main.ll %t/enable.ll > %t1.ll
; RUN: opt < %t0.ll -mtriple=x86_64-linux -passes=instrprof -S | FileCheck %s --check-prefixes=ELF,ELF0
; RUN: opt < %t1.ll -mtriple=x86_64-linux -passes=instrprof -S | FileCheck %s --check-prefixes=ELF,ELF1
; RUN: opt < %t0.ll -mtriple=x86_64-windows -passes=instrprof -S | FileCheck %s --check-prefixes=COFF,COFF0
; RUN: opt < %t1.ll -mtriple=x86_64-windows -passes=instrprof -S | FileCheck %s --check-prefixes=COFF,COFF1

;--- main.ll
declare void @llvm.instrprof.increment(ptr, i64, i32, i32)

$foo_inline = comdat any
$foo_extern = comdat any
$foo_private_profn = comdat any

@__profn_foo_inline = linkonce_odr hidden constant [10 x i8] c"foo_inline"
@__profn_foo_extern = linkonce_odr hidden constant [10 x i8] c"foo_extern"
@__profn_foo_private_profn = private constant [17 x i8] c"foo_private_profn"

;; When value profiling is disabled, __profd_ variables are not referenced by
;; code. We can use private linkage. When enabled, __profd_ needs to be
;; non-local which requires separate comdat on COFF due to a link.exe limitation.

; ELF:   @__profc_foo_inline = linkonce_odr hidden global {{.*}}, section "__llvm_prf_cnts", comdat, align 8
; ELF0:  @__profd_foo_inline = private global {{.*}}, section "__llvm_prf_data", comdat($__profc_foo_inline), align 8
; ELF1:  @__profd_foo_inline = linkonce_odr hidden global {{.*}}, section "__llvm_prf_data", comdat($__profc_foo_inline), align 8
; COFF0: @__profc_foo_inline = linkonce_odr hidden global {{.*}}, section ".lprfc$M", comdat, align 8
; COFF0: @__profd_foo_inline = internal global {{.*}}, section ".lprfd$M", comdat($__profc_foo_inline), align 8
; COFF1: @__profc_foo_inline = linkonce_odr hidden global {{.*}}, section ".lprfc$M", comdat, align 8
; COFF1: @__profd_foo_inline = linkonce_odr hidden global {{.*}}, section ".lprfd$M", comdat, align 8
define weak_odr void @foo_inline() comdat {
  call void @llvm.instrprof.increment(ptr @__profn_foo_inline, i64 0, i32 1, i32 0)
  ret void
}

; ELF:   @__profc_foo_extern = linkonce_odr hidden global{{.*}}, section "__llvm_prf_cnts", comdat, align 8
; ELF0:  @__profd_foo_extern = private global{{.*}}, section "__llvm_prf_data", comdat($__profc_foo_extern), align 8
; ELF1:  @__profd_foo_extern = linkonce_odr hidden global{{.*}}, section "__llvm_prf_data", comdat($__profc_foo_extern), align 8
; COFF:  @__profc_foo_extern = linkonce_odr hidden global{{.*}}, section ".lprfc$M", comdat, align 8
; COFF0: @__profd_foo_extern = internal global{{.*}}, section ".lprfd$M", comdat($__profc_foo_extern), align 8
; COFF1: @__profd_foo_extern = linkonce_odr hidden global{{.*}}, section ".lprfd$M", comdat, align 8
define available_externally void @foo_extern() {
  call void @llvm.instrprof.increment(ptr @__profn_foo_extern, i64 0, i32 1, i32 0)
  ret void
}

; ELF:   @__profc_foo_private_profn = private global{{.*}}, section "__llvm_prf_cnts", comdat, align 8
; ELF0:  @__profd_foo_private_profn = private global{{.*}}, section "__llvm_prf_data", comdat($__profc_foo_private_profn), align 8
; ELF1:  @__profd_foo_private_profn = private global{{.*}}, section "__llvm_prf_data", comdat($__profc_foo_private_profn), align 8
; COFF:  @__profc_foo_private_profn = internal global{{.*}}, section ".lprfc$M", comdat, align 8
; COFF0: @__profd_foo_private_profn = internal global{{.*}}, section ".lprfd$M", comdat($__profc_foo_private_profn), align 8
; COFF1: @__profd_foo_private_profn = internal global{{.*}}, section ".lprfd$M", comdat, align 8
define void @foo_private_profn() comdat {
  call void @llvm.instrprof.increment(ptr @__profn_foo_private_profn, i64 0, i32 1, i32 0)
  ret void
}

; ELF:   @llvm.compiler.used = appending global [3 x ptr] [ptr @__profd_foo_inline, ptr @__profd_foo_extern, ptr @__profd_foo_private_profn]
; COFF0: @llvm.compiler.used = appending global [4 x ptr] [ptr @__llvm_profile_runtime_user, ptr @__profd_foo_inline, ptr @__profd_foo_extern, ptr @__profd_foo_private_profn]
; COFF1: @llvm.used = appending global [5 x ptr] [ptr @__llvm_profile_runtime_user, ptr @__profd_foo_inline, ptr @__profd_foo_extern, ptr @__profd_foo_private_profn, ptr @__llvm_prf_nm]

;--- disable.ll
!llvm.module.flags = !{!0}
!0 = !{i32 2, !"EnableValueProfiling", i32 0}

;--- enable.ll
!llvm.module.flags = !{!0}
!0 = !{i32 2, !"EnableValueProfiling", i32 1}
