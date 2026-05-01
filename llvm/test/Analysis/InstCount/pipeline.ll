; REQUIRES: asserts
; RUN: opt -stats -disable-output -passes='instcount<pre-opt>' < %s 2>&1 | FileCheck %s --check-prefix=PRE
; RUN: opt -stats -disable-output -passes='instcount' < %s 2>&1 | FileCheck %s --check-prefixes=POSTNOOPT
; RUN: opt -stats -disable-output -O0 < %s 2>&1 | FileCheck %s --check-prefixes=PRE,POSTNOOPT
; RUN: opt -stats -disable-output -O3 < %s 2>&1 | FileCheck %s --check-prefixes=PRE,POST
; RUN: opt -stats -disable-output -passes='lto<O3>' < %s 2>&1 | FileCheck %s --check-prefixes=PRE,POST
; RUN: opt -stats -disable-output -passes='lto-pre-link<O3>' < %s 2>&1 | FileCheck %s --check-prefixes=PRE,POST
; RUN: opt -stats -disable-output -passes='thinlto<O3>' < %s 2>&1 | FileCheck %s --check-prefixes=PRE,POST
; RUN: opt -stats -disable-output -passes='thinlto-pre-link<O2>' < %s 2>&1 | FileCheck %s --check-prefixes=PRE,POST

; --- <pre-opt> ---
; PRE-DAG: 4 instcount - Number of basic blocks (before optimizations)
; PRE-DAG: 5 instcount - Number of instructions of all types (before optimizations)
; PRE-DAG: 1 instcount - Number of CondBr insts (before optimizations)

; --- No <pre-opt> in pass but no optimization passes run ---
; POSTNOOPT-DAG: 4 instcount - Number of basic blocks
; POSTNOOPT-DAG: 5 instcount - Number of instructions of all types
; POSTNOOPT-DAG: 1 instcount - Number of CondBr insts

; --- Post optimization values ---
; POST-DAG: 1 instcount - Number of basic blocks
; POST-DAG: 1 instcount - Number of instructions of all types
; POST-NOT: instcount - Number of CondBr insts

define i32 @test_count() {
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
