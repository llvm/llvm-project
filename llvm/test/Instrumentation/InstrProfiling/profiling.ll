; RUN: opt < %s -mtriple=x86_64 -passes=instrprof -S | FileCheck %s --check-prefixes=CHECK,ELF,ELF_GENERIC
; RUN: opt < %s -mtriple=x86_64-linux -passes=instrprof -S | FileCheck %s --check-prefixes=CHECK,ELF_LINUX
; RUN: opt < %s -mtriple=x86_64-apple-macosx10.10.0 -passes=instrprof -S | FileCheck %s --check-prefixes=CHECK,MACHO
; RUN: opt < %s -mtriple=x86_64-windows -passes=instrprof -S | FileCheck %s --check-prefixes=CHECK,WIN

; RUN: opt < %s -mtriple=x86_64-apple-macosx10.10.0 -instrprof -S | FileCheck %s

; ELF_GENERIC: @__llvm_profile_runtime = external global i32
; ELF_LINUX-NOT: @__llvm_profile_runtime
; MACHO: @__llvm_profile_runtime = external global i32
; WIN: @__llvm_profile_runtime = external global i32

@__profn_foo = private constant [3 x i8] c"foo"
; CHECK-NOT: __profn_foo
@__profn_bar = private constant [3 x i8] c"bar"
; CHECK-NOT: __profn_bar
@__profn_baz = private constant [3 x i8] c"baz"
; CHECK-NOT: __profn_baz

; ELF:   @__profc_foo = private global [1 x i64] zeroinitializer, section "__llvm_prf_cnts", comdat, align 8
; ELF:   @__profd_foo = private global { i64, i64, i64, i8*, i8*, i32, [2 x i16] } { i64 [[#]], i64 0, i64 sub (i64 ptrtoint ([1 x i64]* @__profc_foo to i64), i64 ptrtoint ({ i64, i64, i64, i8*, i8*, i32, [2 x i16] }* @__profd_foo to i64)), i8* null, i8* null, i32 1, [2 x i16] zeroinitializer }, section "__llvm_prf_data", comdat($__profc_foo), align 8
; MACHO: @__profc_foo = private global [1 x i64] zeroinitializer, section "__DATA,__llvm_prf_cnts", align 8
; MACHO: @__profd_foo = private {{.*}}, section "__DATA,__llvm_prf_data,regular,live_support", align 8
; WIN:   @__profc_foo = private global [1 x i64] zeroinitializer, section ".lprfc$M", align 8
; WIN:   @__profd_foo = private {{.*}}, section ".lprfd$M", align 8
define void @foo() {
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @__profn_foo, i32 0, i32 0), i64 0, i32 1, i32 0)
  ret void
}

; ELF:   @__profc_bar = private global [1 x i64] zeroinitializer, section "__llvm_prf_cnts", comdat, align 8
; ELF:   @__profd_bar = private {{.*}}, section "__llvm_prf_data", comdat($__profc_bar), align 8
; MACHO: @__profc_bar = private global [1 x i64] zeroinitializer, section "__DATA,__llvm_prf_cnts", align 8
; MACHO: @__profd_bar = private {{.*}}, section "__DATA,__llvm_prf_data,regular,live_support", align 8
; WIN:   @__profc_bar = private global [1 x i64] zeroinitializer, section ".lprfc$M", align 8
; WIN:   @__profd_bar = private {{.*}}, section ".lprfd$M", align 8
define void @bar() {
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @__profn_bar, i32 0, i32 0), i64 0, i32 1, i32 0)
  ret void
}

; ELF:   @__profc_baz = private global [3 x i64] zeroinitializer, section "__llvm_prf_cnts", comdat, align 8
; ELF:   @__profd_baz = private {{.*}}, section "__llvm_prf_data", comdat($__profc_baz), align 8
; MACHO: @__profc_baz = private global [3 x i64] zeroinitializer, section "__DATA,__llvm_prf_cnts", align 8
; MACHO: @__profd_baz = private {{.*}}, section "__DATA,__llvm_prf_data,regular,live_support", align 8
; WIN:   @__profc_baz = private global [3 x i64] zeroinitializer, section ".lprfc$M", align 8
; WIN:   @__profd_baz = private {{.*}}, section ".lprfd$M", align 8
define void @baz() {
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @__profn_baz, i32 0, i32 0), i64 0, i32 3, i32 0)
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @__profn_baz, i32 0, i32 0), i64 0, i32 3, i32 1)
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @__profn_baz, i32 0, i32 0), i64 0, i32 3, i32 2)
  ret void
}

declare void @llvm.instrprof.increment(i8*, i64, i32, i32)

; ELF:   @llvm.compiler.used = appending global {{.*}} @__llvm_profile_runtime {{.*}} @__profd_foo {{.*}} @__profd_bar {{.*}} @__profd_baz
; MACHO: @llvm.used = appending global {{.*}} @__llvm_profile_runtime_user {{.*}} @__profd_foo {{.*}} @__profd_bar {{.*}} @__profd_baz
; WIN:   @llvm.compiler.used = appending global {{.*}} @__llvm_profile_runtime_user {{.*}} @__profd_foo {{.*}} @__profd_bar {{.*}} @__profd_baz

; ELF_GENERIC:      define internal void @__llvm_profile_register_functions() unnamed_addr {
; ELF_GENERIC-NEXT:   call void @__llvm_profile_register_function(i8* bitcast (i32* @__llvm_profile_runtime to i8*))
; ELF_GENERIC-NEXT:   call void @__llvm_profile_register_function(i8* bitcast ({ i64, i64, i64, i8*, i8*, i32, [2 x i16] }* @__profd_foo to i8*))
; ELF_GENERIC-NEXT:   call void @__llvm_profile_register_function(i8* bitcast ({ i64, i64, i64, i8*, i8*, i32, [2 x i16] }* @__profd_bar to i8*))
; ELF_GENERIC-NEXT:   call void @__llvm_profile_register_function(i8* bitcast ({ i64, i64, i64, i8*, i8*, i32, [2 x i16] }* @__profd_baz to i8*))
; ELF_GENERIC-NEXT:   call void @__llvm_profile_register_names_function(i8* getelementptr inbounds {{.*}} @__llvm_prf_nm
; ELF_GENERIC-NEXT:   ret void
; ELF_GENERIC-NEXT: }

; ELF_LINUX-NOT: @__llvm_profile_register_functions()
