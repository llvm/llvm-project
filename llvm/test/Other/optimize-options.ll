;RUN: opt -enable-new-pm=0 -S -O1 -debug-pass=Arguments %s 2>&1 | FileCheck %s
;RUN: opt -enable-new-pm=0 -S -O2 -debug-pass=Arguments %s 2>&1 | FileCheck %s
;RUN: opt -enable-new-pm=0 -S -Os -debug-pass=Arguments %s 2>&1 | FileCheck %s
;RUN: opt -enable-new-pm=0 -S -Oz -debug-pass=Arguments %s 2>&1 | FileCheck %s
;RUN: opt -enable-new-pm=0 -S -O3 -debug-pass=Arguments %s 2>&1 | FileCheck %s

; Just check that we get a non-empty set of passes for each -O option.
;CHECK: Pass Arguments: {{.*}} -print-module
