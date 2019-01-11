; Test to ensure that the Enable Split LTO Unit flag is set properly in the
; summary, and that we correctly silently handle linking bitcode files with
; different values of this flag.

; Linking bitcode both with EnableSplitLTOUnit set should work
; RUN: opt -thinlto-bc -thinlto-split-lto-unit -o %t1 %s
; RUN: llvm-bcanalyzer -dump %t1 | FileCheck %s --check-prefix=SPLITLTOUNIT
; RUN: llvm-dis -o - %t1 | FileCheck %s --check-prefix=ENABLESPLITFLAG
; RUN: opt -thinlto-bc -thinlto-split-lto-unit -o %t2 %s
; RUN: llvm-bcanalyzer -dump %t2 | FileCheck %s --check-prefix=SPLITLTOUNIT
; RUN: llvm-dis -o - %t2 | FileCheck %s --check-prefix=ENABLESPLITFLAG
; RUN: llvm-lto2 run -o %t3 %t1 %t2

; Linking bitcode both without EnableSplitLTOUnit set should work
; RUN: opt -thinlto-bc -thinlto-split-lto-unit=false -o %t1 %s
; RUN: llvm-bcanalyzer -dump %t1 | FileCheck %s --check-prefix=NOSPLITLTOUNIT
; RUN: llvm-dis -o - %t1 | FileCheck %s --check-prefix=NOENABLESPLITFLAG
; RUN: opt -thinlto-bc -thinlto-split-lto-unit=false -o %t2 %s
; RUN: llvm-bcanalyzer -dump %t2 | FileCheck %s --check-prefix=NOSPLITLTOUNIT
; RUN: llvm-dis -o - %t2 | FileCheck %s --check-prefix=NOENABLESPLITFLAG
; RUN: llvm-lto2 run -o %t3 %t1 %t2

; Linking bitcode with different values of EnableSplitLTOUnit should succeed
; (silently skipping any optimizations like whole program devirt that rely
; on all modules being split).
; RUN: opt -thinlto-bc -thinlto-split-lto-unit -o %t1 %s
; RUN: llvm-bcanalyzer -dump %t1 | FileCheck %s --check-prefix=SPLITLTOUNIT
; RUN: llvm-dis -o - %t1 | FileCheck %s --check-prefix=ENABLESPLITFLAG
; RUN: opt -thinlto-bc -thinlto-split-lto-unit=false -o %t2 %s
; RUN: llvm-bcanalyzer -dump %t2 | FileCheck %s --check-prefix=NOSPLITLTOUNIT
; RUN: llvm-dis -o - %t2 | FileCheck %s --check-prefix=NOENABLESPLITFLAG
; RUN: llvm-lto2 run -o %t3 %t1 %t2

; Linking bitcode with different values of EnableSplitLTOUnit (reverse order)
; should succeed (silently skipping any optimizations like whole program devirt
; that rely on all modules being split).
; RUN: opt -thinlto-bc -thinlto-split-lto-unit=false -o %t1 %s
; RUN: llvm-bcanalyzer -dump %t1 | FileCheck %s --check-prefix=NOSPLITLTOUNIT
; RUN: llvm-dis -o - %t1 | FileCheck %s --check-prefix=NOENABLESPLITFLAG
; RUN: opt -thinlto-bc -thinlto-split-lto-unit -o %t2 %s
; RUN: llvm-bcanalyzer -dump %t2 | FileCheck %s --check-prefix=SPLITLTOUNIT
; RUN: llvm-dis -o - %t2 | FileCheck %s --check-prefix=ENABLESPLITFLAG
; RUN: llvm-lto2 run -o %t3 %t1 %t2

; The flag should be set when splitting is disabled (for backwards compatibility
; with older bitcode where it was always enabled).
; SPLITLTOUNIT: <FLAGS op0=8/>
; NOSPLITLTOUNIT: <FLAGS op0=0/>

; Check that the corresponding module flag is set when expected.
; ENABLESPLITFLAG: !{i32 1, !"EnableSplitLTOUnit", i32 1}
; NOENABLESPLITFLAG-NOT: !{i32 1, !"EnableSplitLTOUnit", i32 1}

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
