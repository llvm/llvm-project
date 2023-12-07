;; Check that static counters are allocated for value profiler

; RUN: opt < %s -mtriple=x86_64-unknown-linux -passes=instrprof -vp-static-alloc=true -S | FileCheck %s --check-prefix=STATIC
; RUN: opt < %s -mtriple=powerpc-unknown-linux -passes=instrprof -vp-static-alloc=true -S | FileCheck %s --check-prefix=STATIC
; RUN: opt < %s -mtriple=sparc-unknown-linux -passes=instrprof -vp-static-alloc=true -S | FileCheck %s --check-prefix=STATIC
; RUN: opt < %s -mtriple=s390x-unknown-linux -passes=instrprof -vp-static-alloc=true -S | FileCheck %s --check-prefix=STATIC-EXT
; RUN: opt < %s -mtriple=powerpc64-unknown-linux -passes=instrprof -vp-static-alloc=true -S | FileCheck %s --check-prefix=STATIC-EXT
; RUN: opt < %s -mtriple=sparc64-unknown-linux -passes=instrprof -vp-static-alloc=true -S | FileCheck %s --check-prefix=STATIC-EXT
; RUN: opt < %s -mtriple=mips-unknown-linux -passes=instrprof -vp-static-alloc=true -S | FileCheck %s --check-prefix=STATIC-SEXT
; RUN: opt < %s -mtriple=mips64-unknown-linux -passes=instrprof -vp-static-alloc=true -S | FileCheck %s --check-prefix=STATIC-SEXT
; RUN: opt < %s -mtriple=x86_64-unknown-linux -passes=instrprof -vp-static-alloc=false -S | FileCheck %s --check-prefix=DYN

;; Check that counters have the correct alignments.
; RUN: opt %s -mtriple=powerpc64-unknown-linux -passes=instrprof -S | FileCheck %s --check-prefix=ALIGN
; RUN: opt %s -mtriple=powerpc-ibm-aix -passes=instrprof -S | FileCheck %s --check-prefix=ALIGN
; RUN: opt %s -mtriple=powerpc64-ibm-aix -passes=instrprof -S | FileCheck %s --check-prefix=ALIGN
; RUN: opt %s -mtriple=x86_64-unknown-linux -passes=instrprof -S | FileCheck %s --check-prefix=ALIGN

@__profn_foo = private constant [3 x i8] c"foo"
@__profn_bar = private constant [3 x i8] c"bar"

define i32 @foo(ptr ) {
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 12884901887, i32 1, i32 0)
  %2 = ptrtoint ptr %0 to i64
  call void @llvm.instrprof.value.profile(ptr @__profn_foo, i64 12884901887, i64 %2, i32 0, i32 0)
  %3 = tail call i32 %0()
  ret i32 %3
}

$bar = comdat any

define i32 @bar(ptr ) comdat {
entry:
  call void @llvm.instrprof.increment(ptr @__profn_bar, i64 12884901887, i32 1, i32 0)
  %1 = ptrtoint ptr %0 to i64
  call void @llvm.instrprof.value.profile(ptr @__profn_bar, i64 12884901887, i64 %1, i32 0, i32 0)
  %2 = tail call i32 %0()
  ret i32 %2
}

; Function Attrs: nounwind
declare void @llvm.instrprof.increment(ptr, i64, i32, i32) #0

; Function Attrs: nounwind
declare void @llvm.instrprof.value.profile(ptr, i64, i64, i32, i32) #0

attributes #0 = { nounwind }

; STATIC: @__profvp_foo = private global [1 x i64] zeroinitializer, section "{{[^"]+}}", comdat($__profc_foo)
; STATIC: @__profvp_bar = private global [1 x i64] zeroinitializer, section "{{[^"]+}}", comdat($__profc_bar)
; STATIC: @__llvm_prf_vnodes

; DYN-NOT: @__profvp_foo
; DYN-NOT: @__llvm_prf_vnodes

;; __llvm_prf_vnodes and __llvm_prf_nm are not referenced by other metadata sections.
;; We have to conservatively place them in llvm.used.
; STATIC:      @llvm.used = appending global
; STATIC-SAME:   @__llvm_prf_vnodes
; STATIC-SAME:   @__llvm_prf_nm

; STATIC: call void @__llvm_profile_instrument_target(i64 %3, ptr @__profd_foo, i32 0)
; STATIC-EXT: call void @__llvm_profile_instrument_target(i64 %3, ptr @__profd_foo, i32 zeroext 0)
; STATIC-SEXT: call void @__llvm_profile_instrument_target(i64 %3, ptr @__profd_foo, i32 signext 0)

; STATIC: declare void @__llvm_profile_instrument_target(i64, ptr, i32)
; STATIC-EXT: declare void @__llvm_profile_instrument_target(i64, ptr, i32 zeroext)
; STATIC-SEXT: declare void @__llvm_profile_instrument_target(i64, ptr, i32 signext)

; ALIGN: @__profc_foo = private global {{.*}} section "__llvm_prf_cnts",{{.*}} align 8
; ALIGN: @__profvp_foo = private global {{.*}} section "__llvm_prf_vals",{{.*}} align 8
; ALIGN: @__profd_foo = private global {{.*}} section "__llvm_prf_data",{{.*}} align 8
; ALIGN: @__profc_bar = private global {{.*}} section "__llvm_prf_cnts",{{.*}} align 8
; ALIGN: @__profvp_bar = private global {{.*}} section "__llvm_prf_vals",{{.*}}  align 8
; ALIGN: @__profd_bar = private global {{.*}} section "__llvm_prf_data",{{.*}} align 8
; ALIGN: @__llvm_prf_vnodes = private global {{.*}} section "__llvm_prf_vnds", align 8
; ALIGN: @__llvm_prf_nm = private constant {{.*}} section "__llvm_prf_names", align 1
