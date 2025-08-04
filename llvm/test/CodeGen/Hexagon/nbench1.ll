; RUN: llc -mtriple=hexagon -O3 < %s | FileCheck %s

; if instruction being considered for addition to packet has higher latency,
; end existing packet and start a new one.

; CHECK: .LBB0_4:
; CHECK: p{{[0-3]+}} = cmp.gtu(r{{[0-9]+}},r{{[0-9]+}})
; CHECK-NEXT: }

@array = external dso_local local_unnamed_addr global ptr, align 4

; Function Attrs: nofree norecurse nounwind
define dso_local void @NumSift(i32 %i, i32 %j) local_unnamed_addr #0 {
entry:
  %add36 = shl i32 %i, 1
  %cmp.not37 = icmp ugt i32 %add36, %j
  br i1 %cmp.not37, label %while.end, label %while.body.lr.ph

while.body.lr.ph:                                 ; preds = %entry
  %0 = load ptr, ptr @array, align 4
  %add16 = add i32 %j, 1
  br label %while.body

while.body:                                       ; preds = %while.body.lr.ph, %if.end17
  %add39 = phi i32 [ %add36, %while.body.lr.ph ], [ %add, %if.end17 ]
  %i.addr.038 = phi i32 [ %i, %while.body.lr.ph ], [ %i.addr.1, %if.end17 ]
  %cmp2 = icmp ult i32 %add39, %j
  br i1 %cmp2, label %if.then, label %if.end7

if.then:                                          ; preds = %while.body
  %arrayidx = getelementptr inbounds i32, ptr %0, i32 %add39
  %1 = load i32, ptr %arrayidx, align 4
  %add3 = or i32 %add39, 1
  %arrayidx4 = getelementptr inbounds i32, ptr %0, i32 %add3
  %2 = load i32, ptr %arrayidx4, align 4
  %cmp5 = icmp ult i32 %1, %2
  %spec.select = select i1 %cmp5, i32 %add3, i32 %add39
  br label %if.end7

if.end7:                                          ; preds = %if.then, %while.body
  %k.0 = phi i32 [ %add39, %while.body ], [ %spec.select, %if.then ]
  %arrayidx8 = getelementptr inbounds i32, ptr %0, i32 %i.addr.038
  %3 = load i32, ptr %arrayidx8, align 4
  %arrayidx9 = getelementptr inbounds i32, ptr %0, i32 %k.0
  %4 = load i32, ptr %arrayidx9, align 4
  %cmp10 = icmp ult i32 %3, %4
  br i1 %cmp10, label %if.then11, label %if.end17

if.then11:                                        ; preds = %if.end7
  store i32 %3, ptr %arrayidx9, align 4
  store i32 %4, ptr %arrayidx8, align 4
  br label %if.end17

if.end17:                                         ; preds = %if.end7, %if.then11
  %i.addr.1 = phi i32 [ %k.0, %if.then11 ], [ %add16, %if.end7 ]
  %add = shl i32 %i.addr.1, 1
  %cmp.not = icmp ugt i32 %add, %j
  br i1 %cmp.not, label %while.end, label %while.body

while.end:                                        ; preds = %if.end17, %entry
  ret void
}

attributes #0 = { "target-cpu"="hexagonv65" }
