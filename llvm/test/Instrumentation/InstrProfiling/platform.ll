;; Checks for platform specific section names and initialization code.

; RUN: opt < %s -mtriple=x86_64-apple-macosx10.10.0 -instrprof -S | FileCheck %s -check-prefix=MACHO
; RUN: opt < %s -mtriple=x86_64-apple-macosx10.10.0 -passes=instrprof -S | FileCheck %s -check-prefix=MACHO
; RUN: opt < %s -mtriple=x86_64-unknown-linux -instrprof -S | FileCheck %s -check-prefixes=LINUX,ELF
; RUN: opt < %s -mtriple=x86_64-unknown-linux -passes=instrprof -S | FileCheck %s -check-prefixes=LINUX,ELF
; RUN: opt < %s -mtriple=x86_64-unknown-freebsd -instrprof -S | FileCheck %s -check-prefixes=FREEBSD,ELF
; RUN: opt < %s -mtriple=x86_64-unknown-freebsd -passes=instrprof -S | FileCheck %s -check-prefixes=FREEBSD,ELF
; RUN: opt < %s -mtriple=x86_64-scei-ps4 -instrprof -S | FileCheck %s -check-prefixes=PS4,ELF
; RUN: opt < %s -mtriple=x86_64-scei-ps4 -passes=instrprof -S | FileCheck %s -check-prefixes=PS4,ELF
; RUN: opt < %s -mtriple=x86_64-pc-solaris -instrprof -S | FileCheck %s -check-prefixes=SOLARIS,ELF
; RUN: opt < %s -mtriple=x86_64-pc-solaris -passes=instrprof -S | FileCheck %s -check-prefixes=SOLARIS,ELF
; RUN: opt < %s -mtriple=x86_64-pc-windows -instrprof -S | FileCheck %s -check-prefix=WINDOWS
; RUN: opt < %s -mtriple=x86_64-pc-windows -passes=instrprof -S | FileCheck %s -check-prefix=WINDOWS

@__profn_foo = hidden constant [3 x i8] c"foo"
; MACHO-NOT: __profn_foo
; ELF-NOT: __profn_foo
; WINDOWS-NOT: __profn_foo

; MACHO: @__profc_foo = hidden global [1 x i64] zeroinitializer, section "__DATA,__llvm_prf_cnts", align 8
; ELF: @__profc_foo = hidden global [1 x i64] zeroinitializer, section "__llvm_prf_cnts", align 8
; WINDOWS: @__profc_foo = internal global [1 x i64] zeroinitializer, section ".lprfc$M", align 8

; MACHO: @__profd_foo = hidden {{.*}}, section "__DATA,__llvm_prf_data,regular,live_support", align 8
; ELF: @__profd_foo = hidden {{.*}}, section "__llvm_prf_data", align 8
; WINDOWS: @__profd_foo = internal global {{.*}}, section ".lprfd$M", align 8

; ELF: @__llvm_prf_nm = private constant [{{.*}} x i8] c"{{.*}}", section "{{.*}}__llvm_prf_names", align 1
; WINDOWS: @__llvm_prf_nm = private constant [{{.*}} x i8] c"{{.*}}", section "{{.*}}lprfn$M", align 1

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
; SOLARIS-NOT: define internal void @__llvm_profile_register_functions
; PS4-NOT: define internal void @__llvm_profile_register_functions
; WINDOWS-NOT: define internal void @__llvm_profile_register_functions

;; PR38340: When dynamic registration is used, we had a bug where we'd register
;; something that's not a __profd_* variable.

; MACHO-NOT: define internal void @__llvm_profile_init
; LINUX-NOT: define internal void @__llvm_profile_init
; FREEBSD-NOT: define internal void @__llvm_profile_init
; SOLARIS-NOT: define internal void @__llvm_profile_init
; PS4-NOT: define internal void @__llvm_profile_init
; WINDOWS-NOT: define internal void @__llvm_profile_init
