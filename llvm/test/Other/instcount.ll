; RUN: opt -stats -passes=instcount < %s 2>&1 | FileCheck %s --check-prefix=UNOPT
; RUN: opt -stats -O3 < %s 2>&1 | FileCheck %s --check-prefix=OPT

; --- FIRST RUN (UNOPTIMIZED) ---
; UNOPT-DAG: 8 instcount - Number of Br insts
; UNOPT-DAG: 6 instcount - Number of Call insts
; UNOPT-DAG: 2 instcount - Number of ICmp insts
; UNOPT-DAG: 1 instcount - Number of Ret insts
; UNOPT-DAG: 1 instcount - Number of Switch insts
; UNOPT-DAG: 10 instcount - Number of basic blocks
; UNOPT-DAG: 1 instcount - Number of non-external functions
; UNOPT-DAG: 18 instcount - Number of instructions (of all types)

; --- SECOND RUN (OPTIMIZED) ---
; OPT-DAG: 8 instcount - Number of Br insts
; OPT-DAG: 6 instcount - Number of Call insts
; OPT-DAG: 2 instcount - Number of ICmp insts
; OPT-DAG: 1 instcount - Number of Ret insts
; OPT-DAG: 1 instcount - Number of Switch insts
; OPT-DAG: 10 instcount - Number of basic blocks
; OPT-DAG: 1 instcount - Number of non-external functions
; OPT-DAG: 18 instcount - Number of instructions (of all types)


define dso_local void @foo(i32 noundef %i, i32 noundef %j, i32 noundef %n) {
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
