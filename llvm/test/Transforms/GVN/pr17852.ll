; RUN: opt < %s -passes=gvn
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
%struct.S0 = type { [2 x i8], [2 x i8], [4 x i8], [2 x i8], i32, i32, i32, i32 }
define void @fn1(ptr byval(%struct.S0) align 8 %p1) {
  br label %for.cond
for.cond:                                         ; preds = %1, %0
  br label %for.end
  %f2 = getelementptr inbounds %struct.S0, ptr %p1, i64 0, i32 2
  %f9 = getelementptr inbounds %struct.S0, ptr %p1, i64 0, i32 7
  br label %for.cond
for.end:                                          ; preds = %for.cond
  br i1 true, label %if.else, label %if.then
if.then:                                          ; preds = %for.end
  %f22 = getelementptr inbounds %struct.S0, ptr %p1, i64 0, i32 2
  %f7 = getelementptr inbounds %struct.S0, ptr %p1, i64 0, i32 5
  %tmp7 = load i32, ptr %f7, align 8
  br label %if.end40
if.else:                                          ; preds = %for.end
  br i1 false, label %for.cond18, label %if.then6
if.then6:                                         ; preds = %if.else
  %f3 = getelementptr inbounds %struct.S0, ptr %p1, i64 0, i32 2
  %f5 = getelementptr inbounds %struct.S0, ptr %p1, i64 0, i32 3
  %bf.load13 = load i16, ptr %f5, align 8
  br label %if.end36
for.cond18:                                       ; preds = %if.else
  call void @fn4()
  br i1 true, label %if.end, label %if.end36
if.end:                                           ; preds = %for.cond18
  %f321 = getelementptr inbounds %struct.S0, ptr %p1, i64 0, i32 2
  %f925 = getelementptr inbounds %struct.S0, ptr %p1, i64 0, i32 7
  %f526 = getelementptr inbounds %struct.S0, ptr %p1, i64 0, i32 3
  %bf.load27 = load i16, ptr %f526, align 8
  br label %if.end36
if.end36:                                         ; preds = %if.end, %for.cond18, %if.then6
  %f537 = getelementptr inbounds %struct.S0, ptr %p1, i64 0, i32 3
  %bf.load38 = load i16, ptr %f537, align 8
  %bf.clear39 = and i16 %bf.load38, -16384
  br label %if.end40
if.end40:                                         ; preds = %if.end36, %if.then
  %f6 = getelementptr inbounds %struct.S0, ptr %p1, i64 0, i32 4
  %tmp18 = load i32, ptr %f6, align 4
  call void @fn2(i32 %tmp18)
  %f8 = getelementptr inbounds %struct.S0, ptr %p1, i64 0, i32 6
  %tmp19 = load i32, ptr %f8, align 4
  %tobool41 = icmp eq i32 %tmp19, 0
  br i1 true, label %if.end50, label %if.then42
if.then42:                                        ; preds = %if.end40
  %f547 = getelementptr inbounds %struct.S0, ptr %p1, i64 0, i32 3
  %bf.load48 = load i16, ptr %f547, align 8
  br label %if.end50
if.end50:                                         ; preds = %if.then42, %if.end40
  %f551 = getelementptr inbounds %struct.S0, ptr %p1, i64 0, i32 3
  %bf.load52 = load i16, ptr %f551, align 8
  %bf.clear53 = and i16 %bf.load52, -16384
  ret void
}
declare void @fn2(i32)
declare void @fn4()
