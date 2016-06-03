;; Check that PGO instrumented variables associated with linkonce odr
;; functions are placed in the same comdat section.


; RUN: opt < %s -mtriple=x86_64-apple-macosx10.10.0 -instrprof -S | FileCheck %s
; RUN: opt < %s -mtriple=x86_64-apple-macosx10.10.0 -passes=instrprof -S | FileCheck %s
; RUN: opt < %s -mtriple=x86_64-unknown-linux -instrprof -S | FileCheck %s
; RUN: opt < %s -mtriple=x86_64-unknown-linux -passes=instrprof -S | FileCheck %s
; RUN: opt < %s -mtriple=x86_64-pc-win32-coff -instrprof -S | FileCheck %s --check-prefix=COFF
; RUN: opt < %s -mtriple=x86_64-pc-win32-coff -passes=instrprof -S | FileCheck %s --check-prefix=COFF

$_Z3barIvEvv = comdat any

@__profn__Z3barIvEvv = linkonce_odr hidden constant [11 x i8] c"_Z3barIvEvv", align 1

; CHECK: @__profn__Z3barIvEvv = private constant [11 x i8] c"_Z3barIvEvv", align 1
; CHECK: @__profc__Z3barIvEvv = linkonce_odr hidden global [1 x i64] zeroinitializer, section "{{.*}}__llvm_prf_cnts", comdat($__profv__Z3barIvEvv), align 8
; CHECK: @__profd__Z3barIvEvv = linkonce_odr hidden global { i64, i64, i64*, i8*, i8*, i32, [1 x i16] } { i64 4947693190065689389, i64 0, i64* getelementptr inbounds ([1 x i64], [1 x i64]* @__profc__Z3barIvEvv, i32 0, i32 0), i8*{{.*}}, i8* null, i32 1, [1 x i16] zeroinitializer }, section "{{.*}}__llvm_prf_data", comdat($__profv__Z3barIvEvv), align 8
; CHECK: @__llvm_prf_nm = private constant [{{.*}} x i8] c"{{.*}}", section "{{.*}}__llvm_prf_names"


; COFF: @__profn__Z3barIvEvv = private constant [11 x i8] c"_Z3barIvEvv", align 1
; COFF: @__profc__Z3barIvEvv = linkonce_odr hidden global [1 x i64] zeroinitializer, section "{{.*}}__llvm_prf_cnts", comdat, align 8
; COFF: @__profd__Z3barIvEvv = linkonce_odr hidden global { i64, i64, i64*, i8*, i8*, i32, [1 x i16] } { i64 4947693190065689389, i64 0, i64* getelementptr inbounds ([1 x i64], [1 x i64]* @__profc__Z3barIvEvv, i32 0, i32 0), i8*{{.*}}, i8* null, i32 1, [1 x i16] zeroinitializer }, section "{{.*}}__llvm_prf_data", comdat($__profc__Z3barIvEvv), align 8


declare void @llvm.instrprof.increment(i8*, i64, i32, i32) #1

define linkonce_odr void @_Z3barIvEvv() comdat {
entry:
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([11 x i8], [11 x i8]* @__profn__Z3barIvEvv, i32 0, i32 0), i64 0, i32 1, i32 0)
  ret void
}
