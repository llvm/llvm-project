; RUN: rm -rf %t && split-file %s %t

; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec -mtriple powerpc-ibm-aix-xcoff \
; RUN:     -xcoff-traceback-table=false < %t/no-ref.ll | FileCheck %s --check-prefixes=NOREF
; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec -mtriple powerpc-ibm-aix-xcoff \
; RUN:     -xcoff-traceback-table=false --filetype=obj < %t/no-ref.ll -o %t/no-ref.o
; RUN: llvm-objdump %t/no-ref.o -r | FileCheck %s --check-prefix=NOREF-OBJ

; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec -mtriple powerpc-ibm-aix-xcoff \
; RUN:     -xcoff-traceback-table=false < %t/no-vnds.ll | FileCheck %s --check-prefixes=NOVNDS
; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec -mtriple powerpc-ibm-aix-xcoff \
; RUN:     -xcoff-traceback-table=false --filetype=obj < %t/no-vnds.ll -o %t/no-vnds.o
; RUN: llvm-objdump %t/no-vnds.o -r | FileCheck %s --check-prefix=NOVNDS-OBJ

; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec -mtriple powerpc-ibm-aix-xcoff \
; RUN:     -xcoff-traceback-table=false < %t/with-vnds.ll | FileCheck %s --check-prefixes=WITHVNDS
; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec -mtriple powerpc-ibm-aix-xcoff \
; RUN:     -xcoff-traceback-table=false --filetype=obj < %t/with-vnds.ll -o %t/with-vnds.o
; RUN: llvm-objdump %t/with-vnds.o -tr | FileCheck %s --check-prefix=WITHVNDS-OBJ

; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec -mtriple powerpc-ibm-aix-xcoff \
; RUN:     -xcoff-traceback-table=false < %t/zero-size-cnts-section.ll | FileCheck %s --check-prefixes=ZERO-SIZE-CNTS
; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec -mtriple powerpc-ibm-aix-xcoff \
; RUN:     -xcoff-traceback-table=false --filetype=obj < %t/zero-size-cnts-section.ll -o %t/zero-size-cnts-section.o
; RUN: llvm-objdump %t/zero-size-cnts-section.o -tr | FileCheck %s --check-prefix=ZERO-SIZE-CNTS-OBJ

; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec -mtriple powerpc-ibm-aix-xcoff \
; RUN:     -xcoff-traceback-table=false < %t/zero-size-other-section.ll | FileCheck %s --check-prefixes=ZERO-SIZE-OTHER
; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec -mtriple powerpc64-ibm-aix-xcoff \
; RUN:     -xcoff-traceback-table=false < %t/zero-size-other-section.ll | FileCheck %s --check-prefixes=ZERO-SIZE-OTHER
; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec -mtriple powerpc-ibm-aix-xcoff \
; RUN:     -xcoff-traceback-table=false --filetype=obj < %t/zero-size-other-section.ll -o %t/zero-size-other-section.o
; RUN: llvm-objdump %t/zero-size-other-section.o -tr | FileCheck %s --check-prefix=ZERO-SIZE-OTHER-OBJ
; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec -mtriple powerpc64-ibm-aix-xcoff \
; RUN:     -xcoff-traceback-table=false --filetype=obj < %t/zero-size-other-section.ll -o %t/zero-size-other-section.o
; RUN: llvm-objdump %t/zero-size-other-section.o -tr | FileCheck %s --check-prefix=ZERO-SIZE-OTHER-OBJ


;--- no-ref.ll
; The absence of a __llvm_prf_cnts section should stop generating the .refs.
;
target datalayout = "E-m:a-p:32:32-i64:64-n32"
target triple = "powerpc-ibm-aix7.2.0.0"

@__profd_main = private global i64 zeroinitializer, section "__llvm_prf_data", align 8
@__llvm_prf_nm = private constant [6 x i8] c"\04\00main", section "__llvm_prf_names", align 1

@llvm.used = appending global [2 x ptr]
  [ptr @__profd_main,
   ptr @__llvm_prf_nm], section "llvm.metadata"

define i32 @main() #0 {
entry:
  ret i32 1
}

; NOREF-NOT:  .ref __llvm_prf_data
; NOREF-NOT:  .ref __llvm_prf_names
; NOREF-NOT:  .ref __llvm_prf_vnds

; NOREF-OBJ-NOT: R_REF  __llvm_prf_data
; NOREF-OBJ-NOT: R_REF  __llvm_prf_names
; NOREF-OBJ-NOT: R_REF  __llvm_prf_vnds

;--- no-vnds.ll
; This is the most common case. When -fprofile-generate is used and there exists executable code, we generate the __llvm_prf_cnts, __llvm_prf_data, and __llvm_prf_names sections.
;
target datalayout = "E-m:a-p:32:32-i64:64-n32"
target triple = "powerpc-ibm-aix7.2.0.0"

@__profc_main = private global [1 x i64] zeroinitializer, section "__llvm_prf_cnts", align 8
@__profd_main = private global i64 zeroinitializer, section "__llvm_prf_data", align 8
@__llvm_prf_nm = private constant [6 x i8] c"\04\00main", section "__llvm_prf_names", align 1

@llvm.used = appending global [3 x ptr]
  [ptr @__profc_main,
   ptr @__profd_main,
   ptr @__llvm_prf_nm], section "llvm.metadata"

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

; NOVNDS-OBJ: 00000000 R_REF  __llvm_prf_data
; NOVNDS-OBJ: 00000000 R_REF  __llvm_prf_names
; NOVNDS-OBJ-NOT: R_REF  __llvm_prf_vnds

;--- with-vnds.ll
; When value profiling is needed, the PGO instrumentation generates variables in the __llvm_prf_vnds section, so we generate a .ref for them too.
;
@__profc_main = private global [1 x i64] zeroinitializer, section "__llvm_prf_cnts", align 8
@__profd_main = private global i64 zeroinitializer, section "__llvm_prf_data", align 8
@__llvm_prf_nm = private constant [6 x i8] c"\04\00main", section "__llvm_prf_names", align 1
@__llvm_prf_vnodes = private global [10 x { i64, i64, ptr }] zeroinitializer, section "__llvm_prf_vnds"

@llvm.used = appending global [4 x ptr]
  [ptr @__profc_main,
   ptr @__profd_main,
   ptr @__llvm_prf_nm,
   ptr @__llvm_prf_vnodes], section "llvm.metadata"

define i32 @main() #0 {
entry:
  ret i32 1
}

; WITHVNDS:      .csect __llvm_prf_cnts[RW],3
; WITHVNDS:      .csect __llvm_prf_cnts[RW],3
; WITHVNDS-NEXT: .ref __llvm_prf_data[RW]
; WITHVNDS-NEXT: .ref __llvm_prf_names[RO]
; WITHVNDS-NEXT: .ref __llvm_prf_vnds[RW]

; WITHVNDS-OBJ:      SYMBOL TABLE:
; WITHVNDS-OBJ-NEXT: 00000000      df *DEBUG*	00000000 .file
; WITHVNDS-OBJ-NEXT: 00000000 l       .text	00000008 
; WITHVNDS-OBJ-NEXT: 00000000 g     F .text (csect: ) 	00000000 .main
; WITHVNDS-OBJ-NEXT: 00000008 l       .text	00000006 __llvm_prf_names
; WITHVNDS-OBJ-NEXT: 00000010 l     O .data	00000008 __llvm_prf_cnts
; WITHVNDS-OBJ-NEXT: 00000018 l     O .data	00000008 __llvm_prf_data
; WITHVNDS-OBJ-NEXT: 00000020 l     O .data	000000f0 __llvm_prf_vnds
; WITHVNDS-OBJ-NEXT: 00000110 g     O .data	0000000c main
; WITHVNDS-OBJ-NEXT: 0000011c l       .data	00000000 TOC

; WITHVNDS-OBJ:      RELOCATION RECORDS FOR [.data]:
; WITHVNDS-OBJ-NEXT: OFFSET   TYPE                     VALUE
; WITHVNDS-OBJ-NEXT: 00000000 R_REF                    __llvm_prf_data
; WITHVNDS-OBJ-NEXT: 00000000 R_REF                    __llvm_prf_names
; WITHVNDS-OBJ-NEXT: 00000000 R_REF                    __llvm_prf_vnds
; WITHVNDS-OBJ-NEXT: 00000100 R_POS                    .main
; WITHVNDS-OBJ-NEXT: 00000104 R_POS                    TOC

;--- zero-size-cnts-section.ll
; If __llvm_prf_cnts is of zero size, do not generate the .ref directive.
; The size of the other sections does not matter.

@dummy_cnts = private global [0 x i32] zeroinitializer, section "__llvm_prf_cnts", align 4
@dummy_data = private global [1 x i64] zeroinitializer, section "__llvm_prf_data", align 8
@dummy_name = private constant [0 x i32] zeroinitializer, section "__llvm_prf_names", align 4

@llvm.used = appending global [3 x ptr]
  [ptr @dummy_cnts,
   ptr @dummy_data,
   ptr @dummy_name], section "llvm.metadata"

define i32 @main() #0 {
entry:
  ret i32 1
}

; ZERO-SIZE-CNTS-NOT: .ref __llvm_prf_data[RW]
; ZERO-SIZE-CNTS-NOT: .ref __llvm_prf_names[RO]
; ZERO-SIZE-CNTS-NOT: .ref __llvm_prf_vnds

; ZERO-SIZE-CNTS-OBJ-NOT: R_REF  __llvm_prf_data
; ZERO-SIZE-CNTS-OBJ-NOT: R_REF  __llvm_prf_names
; ZERO-SIZE-CNTS-OBJ-NOT: R_REF  __llvm_prf_vnds

;--- zero-size-other-section.ll
; If __llvm_prf_cnts is of non-zero size, generate the .ref directive even if other sections
; are zero-sized;

@__profc_main = private global [1 x i64] zeroinitializer, section "__llvm_prf_cnts", align 8
@__profd_main = private global [0 x i64] zeroinitializer, section "__llvm_prf_data", align 8
@__llvm_prf_nm = private constant [0 x i8] zeroinitializer, section "__llvm_prf_names", align 1
@__llvm_prf_vnodes = private global [0 x { i64, i64, ptr }] zeroinitializer, section "__llvm_prf_vnds"

@llvm.used = appending global [4 x ptr]
  [ptr @__profc_main,
   ptr @__profd_main,
   ptr @__llvm_prf_nm,
   ptr @__llvm_prf_vnodes], section "llvm.metadata"

define i32 @main() #0 {
entry:
  ret i32 1
}

; ZERO-SIZE-OTHER:      .csect __llvm_prf_cnts[RW],3
; ZERO-SIZE-OTHER:      .csect __llvm_prf_cnts[RW],3
; ZERO-SIZE-OTHER-NEXT: .ref __llvm_prf_data[RW]
; ZERO-SIZE-OTHER-NEXT: .ref __llvm_prf_names[RO]
; ZERO-SIZE-OTHER-NEXT: .ref __llvm_prf_vnds[RW]

; ZERO-SIZE-OTHER-OBJ:      R_REF __llvm_prf_data
; ZERO-SIZE-OTHER-OBJ-NEXT: R_REF __llvm_prf_names
; ZERO-SIZE-OTHER-OBJ-NEXT: R_REF __llvm_prf_vnds

