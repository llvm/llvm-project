; RUN: opt < %s -mtriple=x86_64 -passes=instrprof -S | FileCheck %s --check-prefixes=CHECK,ELF,ELF_GENERIC
; RUN: opt < %s -mtriple=x86_64-linux -passes=instrprof -S | FileCheck %s --check-prefixes=CHECK,ELF_LINUX
; RUN: opt < %s -mtriple=x86_64-apple-macosx10.10.0 -passes=instrprof -S | FileCheck %s --check-prefixes=CHECK,MACHO
; RUN: opt < %s -mtriple=x86_64-windows -passes=instrprof -S | FileCheck %s --check-prefixes=CHECK,WIN

; RUN: opt < %s -mtriple=x86_64-apple-macosx10.10.0 -instrprof -S | FileCheck %s

; ELF_GENERIC: @__llvm_profile_runtime = external global i32
; ELF_LINUX-NOT: @__llvm_profile_runtime
; MACHO: @__llvm_profile_runtime = external global i32
; WIN: @__llvm_profile_runtime = external global i32

@__profn_foo = hidden constant [3 x i8] c"foo"
; CHECK-NOT: __profn_foo
@__profn_bar = hidden constant [4 x i8] c"bar\00"
; CHECK-NOT: __profn_bar
@__profn_baz = hidden constant [3 x i8] c"baz"
; CHECK-NOT: __profn_baz

; ELF:   @__profc_foo = hidden global [1 x i64] zeroinitializer, section "__llvm_prf_cnts", comdat($__profd_foo), align 8
; ELF:   @__profd_foo = hidden {{.*}}, section "__llvm_prf_data", comdat, align 8
; MACHO: @__profc_foo = hidden global [1 x i64] zeroinitializer, section "__DATA,__llvm_prf_cnts", align 8
; MACHO: @__profd_foo = hidden {{.*}}, section "__DATA,__llvm_prf_data,regular,live_support", align 8
; WIN:   @__profc_foo = internal global [1 x i64] zeroinitializer, section ".lprfc$M", align 8
; WIN:   @__profd_foo = internal {{.*}}, section ".lprfd$M", align 8
define void @foo() {
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @__profn_foo, i32 0, i32 0), i64 0, i32 1, i32 0)
  ret void
}

; ELF:   @__profc_bar = hidden global [1 x i64] zeroinitializer, section "__llvm_prf_cnts", comdat($__profd_bar), align 8
; ELF:   @__profd_bar = hidden {{.*}}, section "__llvm_prf_data", comdat, align 8
; MACHO: @__profc_bar = hidden global [1 x i64] zeroinitializer, section "__DATA,__llvm_prf_cnts", align 8
; MACHO: @__profd_bar = hidden {{.*}}, section "__DATA,__llvm_prf_data,regular,live_support", align 8
; WIN:   @__profc_bar = internal global [1 x i64] zeroinitializer, section ".lprfc$M", align 8
; WIN:   @__profd_bar = internal {{.*}}, section ".lprfd$M", align 8
define void @bar() {
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @__profn_bar, i32 0, i32 0), i64 0, i32 1, i32 0)
  ret void
}

; ELF:   @__profc_baz = hidden global [3 x i64] zeroinitializer, section "__llvm_prf_cnts", comdat($__profd_baz), align 8
; ELF:   @__profd_baz = hidden {{.*}}, section "__llvm_prf_data", comdat, align 8
; MACHO: @__profc_baz = hidden global [3 x i64] zeroinitializer, section "__DATA,__llvm_prf_cnts", align 8
; MACHO: @__profd_baz = hidden {{.*}}, section "__DATA,__llvm_prf_data,regular,live_support", align 8
; WIN:   @__profc_baz = internal global [3 x i64] zeroinitializer, section ".lprfc$M", align 8
; WIN:   @__profd_baz = internal {{.*}}, section ".lprfd$M", align 8
define void @baz() {
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @__profn_baz, i32 0, i32 0), i64 0, i32 3, i32 0)
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @__profn_baz, i32 0, i32 0), i64 0, i32 3, i32 1)
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @__profn_baz, i32 0, i32 0), i64 0, i32 3, i32 2)
  ret void
}

declare void @llvm.instrprof.increment(i8*, i64, i32, i32)

; ELF:   @llvm.compiler.used = appending global {{.*}} @__llvm_profile_runtime {{.*}} @__profd_foo {{.*}} @__profd_bar {{.*}} @__profd_baz
; MACHO: @llvm.used = appending global {{.*}} @__llvm_profile_runtime {{.*}} @__profd_foo {{.*}} @__profd_bar {{.*}} @__profd_baz
; WIN:   @llvm.used = appending global {{.*}} @__llvm_profile_runtime {{.*}} @__profd_foo {{.*}} @__profd_bar {{.*}} @__profd_baz

; ELF_GENERIC:      define internal void @__llvm_profile_register_functions() unnamed_addr {
; ELF_GENERIC-NEXT:   call void @__llvm_profile_register_function(i8* bitcast (i32* @__llvm_profile_runtime to i8*))
; ELF_GENERIC-NEXT:   call void @__llvm_profile_register_function(i8* bitcast ({ i64, i64, i64*, i8*, i8*, i32, [2 x i16] }* @__profd_foo to i8*))
; ELF_GENERIC-NEXT:   call void @__llvm_profile_register_function(i8* bitcast ({ i64, i64, i64*, i8*, i8*, i32, [2 x i16] }* @__profd_bar to i8*))
; ELF_GENERIC-NEXT:   call void @__llvm_profile_register_function(i8* bitcast ({ i64, i64, i64*, i8*, i8*, i32, [2 x i16] }* @__profd_baz to i8*))
; ELF_GENERIC-NEXT:   call void @__llvm_profile_register_names_function(i8* getelementptr inbounds {{.*}} @__llvm_prf_nm
; ELF_GENERIC-NEXT:   ret void
; ELF_GENERIC-NEXT: }

; ELF_LINUX-NOT: @__llvm_profile_register_functions()
