;; Checks for platform specific section names and initialization code.

; RUN: opt < %s -mtriple=x86_64-apple-macosx10.10.0 -instrprof -S | FileCheck %s -check-prefix=MACHO
; RUN: opt < %s -mtriple=x86_64-apple-macosx10.10.0 -passes=instrprof -S | FileCheck %s -check-prefix=MACHO
; RUN: opt < %s -mtriple=x86_64-unknown-linux -instrprof -S | FileCheck %s -check-prefix=LINUX
; RUN: opt < %s -mtriple=x86_64-unknown-linux -passes=instrprof -S | FileCheck %s -check-prefix=LINUX
; RUN: opt < %s -mtriple=x86_64-unknown-freebsd -instrprof -S | FileCheck %s -check-prefix=FREEBSD
; RUN: opt < %s -mtriple=x86_64-unknown-freebsd -passes=instrprof -S | FileCheck %s -check-prefix=FREEBSD
; RUN: opt < %s -mtriple=x86_64-scei-ps4 -instrprof -S | FileCheck %s -check-prefix=PS4
; RUN: opt < %s -mtriple=x86_64-scei-ps4 -passes=instrprof -S | FileCheck %s -check-prefix=PS4
; RUN: opt < %s -mtriple=x86_64-pc-solaris -instrprof -S | FileCheck %s -check-prefix=SOLARIS
; RUN: opt < %s -mtriple=x86_64-pc-solaris -passes=instrprof -S | FileCheck %s -check-prefix=SOLARIS

@__profn_foo = hidden constant [3 x i8] c"foo"
; MACHO: @__profn_foo = private constant [3 x i8] c"foo"
; ELF: @__profn_foo = private constant [3 x i8] c"foo"

; MACHO: @__profc_foo = hidden global [1 x i64] zeroinitializer, section "__DATA,__llvm_prf_cnts", align 8
; ELF: @__profc_foo = hidden global [1 x i64] zeroinitializer, section "__llvm_prf_cnts", align 8

; MACHO: @__profd_foo = hidden {{.*}}, section "__DATA,__llvm_prf_data", align 8
; LINUX: @__profd_foo = hidden {{.*}}, section "__llvm_prf_data", align 8
; FREEBSD: @__profd_foo = hidden {{.*}}, section "__llvm_prf_data", align 8
; PS4: @__profd_foo = hidden {{.*}}, section "__llvm_prf_data", align 8
; SOLARIS: @__profd_foo = hidden {{.*}}, section "__llvm_prf_data", align 8

; ELF: @__llvm_prf_nm = private constant [{{.*}} x i8] c"{{.*}}", section "{{.*}}__llvm_prf_names"

define void @foo() {
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @__profn_foo, i32 0, i32 0), i64 0, i32 1, i32 0)
  ret void
}

declare void @llvm.instrprof.increment(i8*, i64, i32, i32)

;; Emit registration functions for platforms that don't find the
;; symbols by their sections.

; MACHO-NOT: define internal void @__llvm_profile_register_functions
; LINUX-NOT: define internal void @__llvm_profile_register_functions
; FREEBSD-NOT: define internal void @__llvm_profile_register_functions
; PS4-NOT: define internal void @__llvm_profile_register_functions
; SOLARIS: define internal void @__llvm_profile_register_functions

; MACHO-NOT: define internal void @__llvm_profile_init
; LINUX-NOT: define internal void @__llvm_profile_init
; FREEBSD-NOT: define internal void @__llvm_profile_init
; PS4-NOT: define internal void @__llvm_profile_init
; SOLARIS: define internal void @__llvm_profile_init
