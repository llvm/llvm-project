;; Checks for platform specific section names and initialization code.

; RUN: opt < %s -mtriple=x86_64-apple-macosx10.10.0 -instrprof -S | FileCheck %s -check-prefix=MACHO
; RUN: opt < %s -mtriple=x86_64-unknown-linux -instrprof -S | FileCheck %s -check-prefix=LINUX
; RUN: opt < %s -mtriple=x86_64-unknown-freebsd -instrprof -S | FileCheck %s -check-prefix=FREEBSD
; RUN: opt < %s -mtriple=x86_64-pc-solaris -instrprof -S | FileCheck %s -check-prefix=SOLARIS

@__llvm_profile_name_foo = hidden constant [3 x i8] c"foo"
; MACHO: @__llvm_profile_name_foo = hidden constant [3 x i8] c"foo", section "__DATA,__llvm_prf_names", align 1
; ELF: @__llvm_profile_name_foo = hidden constant [3 x i8] c"foo", section "__llvm_prf_names", align 1

; MACHO: @__llvm_profile_counters_foo = hidden global [1 x i64] zeroinitializer, section "__DATA,__llvm_prf_cnts", align 8
; ELF: @__llvm_profile_counters_foo = hidden global [1 x i64] zeroinitializer, section "__llvm_prf_cnts", align 8

; MACHO: @__llvm_profile_data_foo = hidden {{.*}}, section "__DATA,__llvm_prf_data", align 8
; LINUX: @__llvm_profile_data_foo = hidden {{.*}}, section "__llvm_prf_data", align 8
; FREEBSD: @__llvm_profile_data_foo = hidden {{.*}}, section "__llvm_prf_data", align 8
; SOLARIS: @__llvm_profile_data_foo = hidden {{.*}}, section "__llvm_prf_data", align 8

define void @foo() {
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @__llvm_profile_name_foo, i32 0, i32 0), i64 0, i32 1, i32 0)
  ret void
}

declare void @llvm.instrprof.increment(i8*, i64, i32, i32)

;; Emit registration functions for platforms that don't find the
;; symbols by their sections.

; MACHO-NOT: define internal void @__llvm_profile_register_functions
; LINUX-NOT: define internal void @__llvm_profile_register_functions
; FREEBSD-NOT: define internal void @__llvm_profile_register_functions
; SOLARIS: define internal void @__llvm_profile_register_functions

; MACHO-NOT: define internal void @__llvm_profile_init
; LINUX-NOT: define internal void @__llvm_profile_init
; FREEBSD-NOT: define internal void @__llvm_profile_init
; SOLARIS: define internal void @__llvm_profile_init
