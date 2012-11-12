;RUN: opt -S -O1 -debug-pass=Arguments 2>&1 | FileCheck %s
;RUN: opt -S -O2 -debug-pass=Arguments 2>&1 | FileCheck %s
;RUN: opt -S -Os -debug-pass=Arguments 2>&1 | FileCheck %s
;RUN: opt -S -Oz -debug-pass=Arguments 2>&1 | FileCheck %s
;RUN: opt -S -O3 -debug-pass=Arguments 2>&1 | FileCheck %s

; Just check that we get a non-empty set of passes for each -O opton.
;CHECK: Pass Arguments: {{.*}} -print-module
