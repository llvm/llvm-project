; RUN: rm -rf %t && split-file %s %t

; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec -mtriple powerpc-ibm-aix-xcoff -xcoff-traceback-table=false < %t/no-ref.ll | FileCheck %s --check-prefixes=NOREF
; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec -mtriple powerpc-ibm-aix-xcoff -xcoff-traceback-table=false < %t/no-vnds.ll | FileCheck %s --check-prefixes=NOVNDS
; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec -mtriple powerpc-ibm-aix-xcoff -xcoff-traceback-table=false < %t/with-vnds.ll | FileCheck %s --check-prefixes=WITHVNDS


;--- no-ref.ll
; The absence of a __llvm_prf_cnts section should stop generating the .refs.
;
target datalayout = "E-m:a-p:32:32-i64:64-n32"
target triple = "powerpc-ibm-aix7.2.0.0"

@__profd_main = private global i64 zeroinitializer, section "__llvm_prf_data", align 8
@__llvm_prf_nm = private constant [6 x i8] c"\04\00main", section "__llvm_prf_names", align 1

@llvm.used = appending global [2 x i8*]
  [i8* bitcast (i64* @__profd_main to i8*),
   i8* getelementptr inbounds ([6 x i8], [6 x i8]* @__llvm_prf_nm, i32 0, i32 0)], section "llvm.metadata"

define i32 @main() #0 {
entry:
  ret i32 1
}

; NOREF-NOT:  .ref __llvm_prf_data
; NOREF-NOT:  .ref __llvm_prf_names
; NOREF-NOT:  .ref __llvm_prf_vnds

;--- no-vnds.ll
; This is the most common case. When -fprofile-generate is used and there exists executable code, we generate the __llvm_prf_cnts, __llvm_prf_data, and __llvm_prf_names sections.
;
target datalayout = "E-m:a-p:32:32-i64:64-n32"
target triple = "powerpc-ibm-aix7.2.0.0"

@__profc_main = private global [1 x i64] zeroinitializer, section "__llvm_prf_cnts", align 8
@__profd_main = private global i64 zeroinitializer, section "__llvm_prf_data", align 8
@__llvm_prf_nm = private constant [6 x i8] c"\04\00main", section "__llvm_prf_names", align 1

@llvm.used = appending global [3 x i8*]
  [i8* bitcast ([1 x i64]* @__profc_main to i8*),
   i8* bitcast (i64* @__profd_main to i8*),
   i8* getelementptr inbounds ([6 x i8], [6 x i8]* @__llvm_prf_nm, i32 0, i32 0)], section "llvm.metadata"

define i32 @main() #0 {
entry:
  ret i32 1
}
; There will be two __llvm_prf_cnts .csects, one to represent the actual csect 
; that holds @__profc_main, and one generated to hold the .ref directives. In 
; XCOFF, a csect can be defined in pieces, so this is is legal assembly.
;
; NOVNDS:      .csect __llvm_prf_cnts[RW],3
; NOVNDS:      .csect __llvm_prf_cnts[RW],3
; NOVNDS-NEXT: .ref __llvm_prf_data[RW]
; NOVNDS-NEXT: .ref __llvm_prf_names[RO]
; NOVNDS-NOT:  .ref __llvm_prf_vnds

;--- with-vnds.ll
; When value profiling is needed, the PGO instrumentation generates variables in the __llvm_prf_vnds section, so we generate a .ref for them too.
;
@__profc_main = private global [1 x i64] zeroinitializer, section "__llvm_prf_cnts", align 8
@__profd_main = private global i64 zeroinitializer, section "__llvm_prf_data", align 8
@__llvm_prf_nm = private constant [6 x i8] c"\04\00main", section "__llvm_prf_names", align 1
@__llvm_prf_vnodes = private global [10 x { i64, i64, i8* }] zeroinitializer, section "__llvm_prf_vnds"

@llvm.used = appending global [4 x i8*]
  [i8* bitcast ([1 x i64]* @__profc_main to i8*),
   i8* bitcast (i64* @__profd_main to i8*),
   i8* getelementptr inbounds ([6 x i8], [6 x i8]* @__llvm_prf_nm, i32 0, i32 0),
   i8* bitcast ([10 x { i64, i64, i8* }]* @__llvm_prf_vnodes to i8*)], section "llvm.metadata"

define i32 @main() #0 {
entry:
  ret i32 1
}

; WITHVNDS:      .csect __llvm_prf_cnts[RW],3
; WITHVNDS:      .csect __llvm_prf_cnts[RW],3
; WITHVNDS-NEXT: .ref __llvm_prf_data[RW]
; WITHVNDS-NEXT: .ref __llvm_prf_names[RO]
; WITHVNDS-NEXT: .ref __llvm_prf_vnds[RW]
