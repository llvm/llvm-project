; Testing with all of the below run lines that the pass gets added to the appropriate pipelines
; REQUIRES: asserts
; RUN: opt -stats -enable-detailed-function-properties -disable-output -passes=func-properties-stats < %s 2>&1 | FileCheck %s
; RUN: opt -stats -enable-detailed-function-properties -disable-output -passes='thinlto<O3>'< %s 2>&1 | FileCheck %s
; RUN: opt -stats -enable-detailed-function-properties -disable-output -passes='thinlto-pre-link<O2>' < %s 2>&1 | FileCheck %s
; RUN: opt -stats -enable-detailed-function-properties -disable-output -passes='lto<O1>' < %s 2>&1 | FileCheck %s
; RUN: opt -stats -enable-detailed-function-properties -disable-output -O3 < %s 2>&1 | FileCheck %s
; RUN: opt -stats -enable-detailed-function-properties -disable-output -O0 < %s 2>&1 | FileCheck %s

; CHECK-DAG: 10 func-properties-stats - Number of basic blocks
; CHECK-DAG: 8 func-properties-stats - Number of branch instructions
; CHECK-DAG: 10 func-properties-stats - Number of branch successors
; CHECK-DAG: 2 func-properties-stats - Number of conditional branch instructions
; CHECK-DAG: 6 func-properties-stats - Number of unconditional branch instructions
; CHECK-DAG: 18 func-properties-stats - Number of instructions (of all types)
; CHECK-DAG: 14 func-properties-stats - Number of basic block successors
; CHECK-DAG: 1 func-properties-stats - Number of switch instructions
; CHECK-DAG: 4 func-properties-stats - Number of switch successors


define void @foo(i32 %i, i32 %j, i32 %n) {
entry:
  %cmp = icmp slt i32 %i, %j
  br i1 %cmp, label %if.then, label %if.end

if.then:
  call void @f()
  br label %if.end

if.end:
  switch i32 %i, label %sw.default [
    i32 1, label %sw.bb
    i32 2, label %sw.bb1
    i32 3, label %sw.bb1
  ]

sw.bb:
  call void @g()
  br label %sw.epilog

sw.bb1:
  call void @h()
  br label %sw.epilog

sw.default:
  call void @k()
  br label %sw.epilog

sw.epilog:
  %cmp2 = icmp sgt i32 %i, %n
  br i1 %cmp2, label %if.then3, label %if.else

if.then3:
  call void @l()
  br label %if.end4

if.else:
  call void @m()
  br label %if.end4

if.end4:
  ret void
}

declare void @f()
declare void @g()
declare void @h()
declare void @k()
declare void @l()
declare void @m()
