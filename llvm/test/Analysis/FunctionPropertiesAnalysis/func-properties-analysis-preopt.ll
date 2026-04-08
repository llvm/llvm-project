; REQUIRES: asserts
; RUN: opt -stats -enable-detailed-function-properties -disable-output -passes='func-properties-stats<pre-opt>' < %s 2>&1 | FileCheck %s --check-prefix=PRE
; RUN: opt -stats -enable-detailed-function-properties -disable-output -passes='func-properties-stats' < %s 2>&1 | FileCheck %s --check-prefixes=POSTNOOPT
; RUN: opt -stats -enable-detailed-function-properties -disable-output -O0 < %s 2>&1 | FileCheck %s --check-prefixes=PRE,POSTNOOPT
; RUN: opt -stats -enable-detailed-function-properties -disable-output -O3 < %s 2>&1 | FileCheck %s --check-prefixes=PRE,POST
; RUN: opt -stats -enable-detailed-function-properties -disable-output -passes='thinlto<O3>' < %s 2>&1 | FileCheck %s --check-prefixes=PRE,POST
; RUN: opt -stats -enable-detailed-function-properties -disable-output -passes='thinlto-pre-link<O2>' < %s 2>&1 | FileCheck %s --check-prefixes=PRE,POST

; --- <pre-opt> ---
; PRE-DAG: 4 func-properties-stats - Number of basic blocks (before optimizations)
; PRE-DAG: 5 func-properties-stats - Number of instructions (of all types) (before optimizations)
; PRE-DAG: 4 func-properties-stats - Number of basic block successors (before optimizations)

; --- No <pre-opt> in pass but no optimizations ---
; POSTNOOPT-DAG: 4 func-properties-stats - Number of basic blocks (after optimizations)
; POSTNOOPT-DAG: 5 func-properties-stats - Number of instructions (of all types) (after optimizations)
; POSTNOOPT-DAG: 4 func-properties-stats - Number of basic block successors (after optimizations)

; --- Post optimization values ---
; POST-DAG: 1 func-properties-stats - Number of basic blocks (after optimizations)
; POST-DAG: 1 func-properties-stats - Number of instructions (of all types) (after optimizations)
; POST-NOT: func-properties-stats - Number of basic block successors (after optimizations)

define i32 @test_cfg() {
entry:
  ; This branch is trivially resolvable
  br i1 true, label %then, label %else

then:
  br label %end

else:
  br label %end

end:
  %phi = phi i32 [ 1, %then ], [ 2, %else ]
  ret i32 %phi
}
