;; Test runtime symbols and various linkages.

; RUN: opt < %s -mtriple=x86_64-apple-macosx10.10.0 -passes=instrprof -S | FileCheck %s --check-prefixes=MACHO
; RUN: opt < %s -mtriple=x86_64 -passes=instrprof -S | FileCheck %s --check-prefix=ELF_GENERIC
; RUN: opt < %s -mtriple=x86_64-unknown-linux -passes=instrprof -S | FileCheck %s --check-prefixes=ELF,ELFRT
; RUN: opt < %s -mtriple=x86_64-unknown-fuchsia -passes=instrprof -S | FileCheck %s --check-prefixes=ELF,ELFRT
; RUN: opt < %s -mtriple=x86_64-scei-ps4 -passes=instrprof -S | FileCheck %s --check-prefixes=ELF,PS
; RUN: opt < %s -mtriple=x86_64-sie-ps5 -passes=instrprof -S | FileCheck %s --check-prefixes=ELF,PS
; RUN: opt < %s  -mtriple=x86_64-pc-win32-coff -passes=instrprof -S | FileCheck %s --check-prefixes=COFF
; RUN: opt < %s -mtriple=powerpc64-ibm-aix-xcoff -passes=instrprof -S | FileCheck %s --check-prefixes=XCOFF

; MACHO: @__llvm_profile_runtime = external global i32
; ELF_GENERIC: @__llvm_profile_runtime = external global i32
; ELF-NOT: @__llvm_profile_runtime = external global i32
; XCOFF: @__llvm_profile_runtime = external global i32

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
; XCOFF: @__profc_foo = private global
; XCOFF-NOT: comdat
; XCOFF: @__profd_foo = private global
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
; XCOFF: @__profc_foo_weak = private global
; XCOFF: @__profd_foo_weak = private global
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
; XCOFF: @"__profc_linkage.ll:foo_internal" = private global
; XCOFF: @"__profd_linkage.ll:foo_internal" = private global
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
; XCOFF: @__profc_foo_inline = private global
; XCOFF: @__profd_foo_inline = private global
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
; XCOFF: @__profc_foo_extern = private global
; XCOFF: @__profd_foo_extern = private global
define available_externally void @foo_extern() {
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([10 x i8], [10 x i8]* @__profn_foo_extern, i32 0, i32 0), i64 0, i32 1, i32 0)
  ret void
}

declare void @llvm.instrprof.increment(i8*, i64, i32, i32)

; ELF:         @llvm.compiler.used = appending global {{.*}} @__profd_foo {{.*}}
; ELF_GENERIC: @llvm.compiler.used = appending global {{.*}} @__llvm_profile_runtime {{.*}} @__profd_foo {{.*}}
; MACHO:       @llvm.compiler.used = appending global {{.*}} @__llvm_profile_runtime_user {{.*}} @__profd_foo {{.*}}
; COFF:        @llvm.compiler.used = appending global {{.*}} @__llvm_profile_runtime_user {{.*}} @__profd_foo {{.*}}
; XCOFF:       @llvm.used = appending global {{.*}} @__llvm_profile_runtime_user {{.*}} @__profd_foo {{.*}}

; MACHO: define linkonce_odr hidden i32 @__llvm_profile_runtime_user() {{.*}} {
; MACHO:   %[[REG:.*]] = load i32, i32* @__llvm_profile_runtime
; MACHO:   ret i32 %[[REG]]
; MACHO: }
; COFF: define linkonce_odr hidden i32 @__llvm_profile_runtime_user() {{.*}} comdat {
; ELFRT-NOT: define linkonce_odr hidden i32 @__llvm_profile_runtime_user() {{.*}} {
; ELFRT-NOT:   %[[REG:.*]] = load i32, i32* @__llvm_profile_runtime
; PS: define linkonce_odr hidden i32 @__llvm_profile_runtime_user() {{.*}} {
; PS:   %[[REG:.*]] = load i32, i32* @__llvm_profile_runtime
; XCOFF: define linkonce_odr hidden i32 @__llvm_profile_runtime_user() {{.*}} {
; XCOFF:   %[[REG:.*]] = load i32, i32* @__llvm_profile_runtime
; XCOFF:   ret i32 %[[REG]]
; XCOFF: }

; ELF_GENERIC:      define internal void @__llvm_profile_register_functions() unnamed_addr {
; ELF_GENERIC-NEXT:   call void @__llvm_profile_register_function(i8* bitcast (i32* @__llvm_profile_runtime to i8*))
; ELF_GENERIC-NEXT:   call void @__llvm_profile_register_function(i8* bitcast ({ i64, i64, i64, i8*, i8*, i32, [{{.*}} x i16] }* @__profd_foo to i8*))
; ELF_GENERIC-NEXT:   call void @__llvm_profile_register_function(i8* bitcast ({ i64, i64, i64, i8*, i8*, i32, [{{.*}} x i16] }* @__profd_foo_weak to i8*))
; ELF_GENERIC:        call void @__llvm_profile_register_names_function(i8* getelementptr inbounds {{.*}} @__llvm_prf_nm
; ELF_GENERIC-NEXT:   ret void
; ELF_GENERIC-NEXT: }

; XCOFF-NOT:  internal void @__llvm_profile_register_functions() 
