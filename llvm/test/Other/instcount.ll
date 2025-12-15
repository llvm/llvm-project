; REQUIRES: asserts
; RUN: opt -stats -passes=instcount -disable-output < %s 2>&1 | FileCheck %s
; RUN: opt -stats -passes='thinlto<O3>' -disable-output < %s 2>&1 | FileCheck %s
; RUN: opt -stats -passes='thinlto-pre-link<O2>' -disable-output < %s 2>&1 | FileCheck %s
; RUN: opt -stats -passes='lto<O1>' -disable-output < %s 2>&1 | FileCheck %s
; RUN: opt -stats -passes='lto-pre-link<Os>' -disable-output < %s 2>&1 | FileCheck %s
; RUN: opt -stats -O3 -disable-output < %s 2>&1 | FileCheck %s
; RUN: opt -stats -O0 -disable-output < %s 2>&1 | FileCheck %s

; CHECK-DAG: 8 instcount - Number of Br insts
; CHECK-DAG: 6 instcount - Number of Call insts
; CHECK-DAG: 2 instcount - Number of ICmp insts
; CHECK-DAG: 1 instcount - Number of Ret insts
; CHECK-DAG: 1 instcount - Number of Switch insts
; CHECK-DAG: 10 instcount - Number of basic blocks
; CHECK-DAG: 1 instcount - Number of non-external functions
; CHECK-DAG: 18 instcount - Number of instructions (of all types)


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
