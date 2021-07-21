;; Check that runtime symbols get appropriate linkage.

; RUN: opt < %s -mtriple=x86_64-apple-macosx10.10.0 -instrprof -S | FileCheck %s --check-prefixes=MACHO
; RUN: opt < %s -mtriple=x86_64-unknown-linux -instrprof -S | FileCheck %s --check-prefixes=ELF
; RUN: opt < %s -mtriple=x86_64-unknown-fuchsia -instrprof -S | FileCheck %s --check-prefixes=ELF
; RUN: opt < %s  -mtriple=x86_64-pc-win32-coff -instrprof -S | FileCheck %s --check-prefixes=COFF
; RUN: opt < %s -mtriple=x86_64-apple-macosx10.10.0 -passes=instrprof -S | FileCheck %s --check-prefixes=MACHO
; RUN: opt < %s -mtriple=x86_64-unknown-linux -passes=instrprof -S | FileCheck %s --check-prefixes=ELF
; RUN: opt < %s -mtriple=x86_64-unknown-fuchsia -passes=instrprof -S | FileCheck %s --check-prefixes=ELF
; RUN: opt < %s  -mtriple=x86_64-pc-win32-coff -passes=instrprof -S | FileCheck %s --check-prefixes=COFF

; MACHO: @__llvm_profile_runtime = external global i32
; ELF-NOT: @__llvm_profile_runtime = external global i32

; ELF: $__profc_foo = comdat nodeduplicate
; ELF: $__profc_foo_weak = comdat nodeduplicate
; ELF: $"__profc_linkage.ll:foo_internal" = comdat nodeduplicate
; ELF: $__profc_foo_inline = comdat nodeduplicate
; ELF: $__profc_foo_extern = comdat any

@__profn_foo = private constant [3 x i8] c"foo"
@__profn_foo_weak = weak hidden constant [8 x i8] c"foo_weak"
@"__profn_linkage.ll:foo_internal" = private constant [23 x i8] c"linkage.ll:foo_internal"
@__profn_foo_inline = linkonce_odr hidden constant [10 x i8] c"foo_inline"
@__profn_foo_extern = linkonce_odr hidden constant [10 x i8] c"foo_extern"

; ELF: @__profc_foo = private global {{.*}} section "__llvm_prf_cnts", comdat
; ELF: @__profd_foo = private global {{.*}} section "__llvm_prf_data", comdat($__profc_foo)
; MACHO: @__profc_foo = private global
; MACHO: @__profd_foo = private global
; COFF: @__profc_foo = private global
; COFF-NOT: comdat
; COFF: @__profd_foo = private global
define void @foo() {
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @__profn_foo, i32 0, i32 0), i64 0, i32 1, i32 0)
  ret void
}

; ELF: @__profc_foo_weak = weak hidden global{{.*}}section "__llvm_prf_cnts", comdat, align 8
; ELF: @__profd_foo_weak = private global{{.*}}section "__llvm_prf_data", comdat($__profc_foo_weak)
; MACHO: @__profc_foo_weak = weak hidden global
; MACHO: @__profd_foo_weak = weak hidden global
; COFF: @__profc_foo_weak = weak hidden global
; COFF: @__profd_foo_weak = private global
define weak void @foo_weak() {
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([8 x i8], [8 x i8]* @__profn_foo_weak, i32 0, i32 0), i64 0, i32 1, i32 0)
  ret void
}

; ELF: @"__profc_linkage.ll:foo_internal" = private global{{.*}}section "__llvm_prf_cnts", comdat, align 8
; ELF: @"__profd_linkage.ll:foo_internal" = private global{{.*}}section "__llvm_prf_data", comdat($"__profc_linkage.ll:foo_internal"), align 8
; MACHO: @"__profc_linkage.ll:foo_internal" = private global
; MACHO: @"__profd_linkage.ll:foo_internal" = private global
; COFF: @"__profc_linkage.ll:foo_internal" = private global
; COFF: @"__profd_linkage.ll:foo_internal" = private global
define internal void @foo_internal() {
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @"__profn_linkage.ll:foo_internal", i32 0, i32 0), i64 0, i32 1, i32 0)
  ret void
}

; ELF: @__profc_foo_inline = linkonce_odr hidden global{{.*}}section "__llvm_prf_cnts", comdat, align 8
; ELF: @__profd_foo_inline = private global{{.*}}section "__llvm_prf_data", comdat($__profc_foo_inline), align 8
; MACHO: @__profc_foo_inline = linkonce_odr hidden global
; MACHO: @__profd_foo_inline = linkonce_odr hidden global
; COFF: @__profc_foo_inline = linkonce_odr hidden global{{.*}} section ".lprfc$M", align 8
; COFF: @__profd_foo_inline = private global{{.*}} section ".lprfd$M", align 8
define linkonce_odr void @foo_inline() {
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([10 x i8], [10 x i8]* @__profn_foo_inline, i32 0, i32 0), i64 0, i32 1, i32 0)
  ret void
}

; ELF: @__profc_foo_extern = linkonce_odr hidden global {{.*}}section "__llvm_prf_cnts", comdat, align 8
; ELF: @__profd_foo_extern = private global {{.*}}section "__llvm_prf_data", comdat($__profc_foo_extern), align 8
; MACHO: @__profc_foo_extern = linkonce_odr hidden global
; MACHO: @__profd_foo_extern = linkonce_odr hidden global
; COFF: @__profc_foo_extern = linkonce_odr hidden global {{.*}}section ".lprfc$M", comdat, align 8
; COFF: @__profd_foo_extern = private global {{.*}}section ".lprfd$M", comdat($__profc_foo_extern), align 8
define available_externally void @foo_extern() {
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([10 x i8], [10 x i8]* @__profn_foo_extern, i32 0, i32 0), i64 0, i32 1, i32 0)
  ret void
}

declare void @llvm.instrprof.increment(i8*, i64, i32, i32)

; MACHO: define linkonce_odr hidden i32 @__llvm_profile_runtime_user() {{.*}} {
; MACHO:   %[[REG:.*]] = load i32, i32* @__llvm_profile_runtime
; MACHO:   ret i32 %[[REG]]
; MACHO: }
; COFF: define linkonce_odr hidden i32 @__llvm_profile_runtime_user() {{.*}} comdat {
; ELF-NOT: define linkonce_odr hidden i32 @__llvm_profile_runtime_user() {{.*}} {
; ELF-NOT:   %[[REG:.*]] = load i32, i32* @__llvm_profile_runtime
