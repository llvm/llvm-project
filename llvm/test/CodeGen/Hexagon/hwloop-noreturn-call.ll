; RUN: llc -march=hexagon < %s | FileCheck %s

target triple = "hexagon"

; CHECK-LABEL: danny:
; CHECK-DAG: loop0
; CHECK-DAG: call trap
define void @danny(ptr %p, i32 %n, i32 %k) #0 {
entry:
  br label %for.body

for.body:                                         ; preds = %entry
  %t0 = phi i32 [ 0, %entry ], [ %t1, %for.cont ]
  %t1 = add i32 %t0, 1
  %t2 = getelementptr i32, ptr %p, i32 %t0
  store i32 %t1, ptr %t2, align 4
  %c = icmp sgt i32 %t1, %k
  br i1 %c, label %noret, label %for.cont

for.cont:
  %cmp = icmp slt i32 %t0, %n
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.cond
  ret void

noret:
  call void @trap() #1
  br label %for.cont
}

; CHECK-LABEL: sammy:
; CHECK-DAG: loop0
; CHECK-DAG: callr
define void @sammy(ptr %p, i32 %n, i32 %k, ptr %f) #0 {
entry:
  br label %for.body

for.body:                                         ; preds = %entry
  %t0 = phi i32 [ 0, %entry ], [ %t1, %for.cont ]
  %t1 = add i32 %t0, 1
  %t2 = getelementptr i32, ptr %p, i32 %t0
  store i32 %t1, ptr %t2, align 4
  %c = icmp sgt i32 %t1, %k
  br i1 %c, label %noret, label %for.cont

for.cont:
  %cmp = icmp slt i32 %t0, %n
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.cond
  ret void

noret:
  call void (...) %f() #1
  br label %for.cont
}

declare void @trap() #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length64b" }
attributes #1 = { nounwind noreturn }

