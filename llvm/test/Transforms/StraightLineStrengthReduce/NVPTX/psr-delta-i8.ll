; RUN: opt < %s -passes=slsr -enable-partial-strength-reduce -S | FileCheck %s

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-unknown-unknown"

; CHECK-LABEL: psr_i8_zero_delta(
; CHECK-SAME:      ptr [[IN:%.*]], ptr [[OUT:%.*]], i64 [[ADD:%.*]], i1 [[CMP0:%.*]], i1 [[CMP1:%.*]], i1 [[CMP2:%.*]])
; CHECK-NEXT: [[ENTRY:.*]]:
; CHECK-NEXT:   [[GEP:%.*]] = getelementptr  i8, ptr [[IN]], i64 [[ADD]]
; CHECK-NEXT:   br i1 [[CMP0]], label %[[IF0:.*]], label %[[ENDIF0:.*]]
; CHECK-EMPTY:
; CHECK-NEXT: [[IF0]]:
; CHECK-NEXT:   [[GEP0:%.*]] = getelementptr inbounds i8, ptr [[GEP]], i64 32
; CHECK-NEXT:   [[LOAD0:%.*]] = load i8, ptr [[GEP0]]
; CHECK-NEXT:   br label %[[ENDIF0]]
; CHECK-EMPTY:
; CHECK-NEXT: [[ENDIF0]]:
; CHECK-NEXT:   [[VAL0:%.*]] = phi i8 [ 0, %[[ENTRY]] ], [ [[LOAD0]], %[[IF0]] ]
; CHECK-NEXT:   br i1 [[CMP1]], label %[[IF1:.*]], label %[[ENDIF1:.*]]
; CHECK-EMPTY:
; CHECK-NEXT: [[IF1]]:
; CHECK-NEXT:   [[GEP1:%.*]] = getelementptr inbounds i8, ptr [[GEP]], i64 64
; CHECK-NEXT:   [[LOAD1:%.*]] = load i8, ptr [[GEP1]]
; CHECK-NEXT:   br label %[[ENDIF1]]
; CHECK-EMPTY:
; CHECK-NEXT: [[ENDIF1]]:
; CHECK-NEXT:   [[VAL1:%.*]] = phi i8 [ 0, %[[ENDIF0]] ], [ [[LOAD1]], %[[IF1]] ]
; CHECK-NEXT:   br i1 [[CMP2]], label %[[IF2:.*]], label %[[ENDIF2:.*]]
; CHECK-EMPTY:
; CHECK-NEXT: [[IF2]]:
; CHECK-NEXT:   [[GEP2:%.*]] = getelementptr inbounds i8, ptr [[GEP]], i64 96
; CHECK-NEXT:   [[LOAD2:%.*]] = load i8, ptr [[GEP2]]
; CHECK-NEXT:   br label %[[ENDIF2]]
; CHECK-EMPTY:
; CHECK-NEXT: [[ENDIF2]]:
define void @psr_i8_zero_delta(ptr %in, ptr %out, i64 %add, i1 %cmp0, i1 %cmp1, i1 %cmp2) {
entry:
  br i1 %cmp0, label %if.0, label %endif.0

if.0:
  %getElem0.0 = getelementptr inbounds i8, ptr %in, i64 %add
  %getElem0.1 = getelementptr inbounds i8, ptr %getElem0.0, i64 32
  %load0 = load i8, ptr %getElem0.1
  br label %endif.0

endif.0:
  %value0 = phi i8 [ 0, %entry ], [ %load0, %if.0 ]
  br i1 %cmp1, label %if.1, label %endif.1

if.1:
  %getElem1.0 = getelementptr inbounds i8, ptr %in, i64 %add
  %getElem1.1 = getelementptr inbounds i8, ptr %getElem1.0, i64 64
  %load1 = load i8, ptr %getElem1.1
  br label %endif.1

endif.1:
  %value1 = phi i8 [ 0, %endif.0 ], [ %load1, %if.1 ]
  br i1 %cmp2, label %if.2, label %endif.2

if.2:
  %getElem2.0 = getelementptr inbounds i8, ptr %in, i64 %add
  %getElem2.1 = getelementptr inbounds i8, ptr %getElem2.0, i64 96
  %load2 = load i8, ptr %getElem2.1
  br label %endif.2

endif.2:
  %value2 = phi i8 [ 0, %endif.1 ], [ %load2, %if.2 ]
  %out0 = add i8 %value0, %value1
  %out1 = add i8 %out0, %value2
  store i8 %out1, ptr %out

  ret void
}

; CHECK-LABEL: psr_i8_zero_delta_2(
; CHECK-SAME:      ptr [[IN:%.*]], ptr [[OUT:%.*]], i64 [[ADD:%.*]], i1 [[CMP0:%.*]], i1 [[CMP1:%.*]], i1 [[CMP2:%.*]])
; CHECK-NEXT: [[ENTRY:.*]]:
; CHECK-NEXT:   [[GEP0:%.*]] = getelementptr i8, ptr [[IN]], i64 [[ADD]]
; CHECK-NEXT:   br i1 [[CMP0]], label %[[IF0:.*]], label %[[ENDIF0:.*]]
; CHECK-EMPTY:
; CHECK-NEXT: [[IF0]]:
; CHECK-NEXT:   [[LOAD0:%.*]] = load i8, ptr [[GEP0]]
; CHECK-NEXT:   br label %[[ENDIF0]]
; CHECK-EMPTY:
; CHECK-NEXT: [[ENDIF0]]:
; CHECK-NEXT:   [[VAL0:%.*]] = phi i8 [ 0, %[[ENTRY]] ], [ [[LOAD0]], %[[IF0]] ]
; CHECK-NEXT:   br i1 [[CMP1]], label %[[IF1:.*]], label %[[ENDIF1:.*]]
; CHECK-EMPTY:
; CHECK-NEXT: [[IF1]]:
; CHECK-NEXT:   [[GEP1:%.*]] = getelementptr inbounds i8, ptr [[GEP0]], i64 32
; CHECK-NEXT:   [[LOAD1:%.*]] = load i8, ptr [[GEP1]]
; CHECK-NEXT:   br label %[[ENDIF1]]
; CHECK-EMPTY:
; CHECK-NEXT: [[ENDIF1]]:
; CHECK-NEXT:   [[VAL1:%.*]] = phi i8 [ 0, %[[ENDIF0]] ], [ [[LOAD1]], %[[IF1]] ]
; CHECK-NEXT:   br i1 [[CMP2]], label %[[IF2:.*]], label %[[ENDIF2:.*]]
; CHECK-EMPTY:
; CHECK-NEXT: [[IF2]]:
; CHECK-NEXT:   [[GEP2:%.*]] = getelementptr inbounds i8, ptr [[GEP0]], i64 64
; CHECK-NEXT:   [[LOAD2:%.*]] = load i8, ptr [[GEP2]]
; CHECK-NEXT:   br label %[[ENDIF2]]
; CHECK-EMPTY:
; CHECK-NEXT: [[ENDIF2]]:

define void @psr_i8_zero_delta_2(ptr %in, ptr %out, i64 %add, i1 %cmp0, i1 %cmp1, i1 %cmp2) {
entry:
  br i1 %cmp0, label %if.0, label %endif.0

if.0:
  %getElem0.0 = getelementptr inbounds i8, ptr %in, i64 %add
  %load0 = load i8, ptr %getElem0.0
  br label %endif.0

endif.0:
  %value0 = phi i8 [ 0, %entry ], [ %load0, %if.0 ]
  br i1 %cmp1, label %if.1, label %endif.1

if.1:
  %getElem1.0 = getelementptr i8, ptr %in, i64 %add
  %getElem1.1 = getelementptr inbounds i8, ptr %getElem1.0, i64 32
  %load1 = load i8, ptr %getElem1.1
  br label %endif.1

endif.1:
  %value1 = phi i8 [ 0, %endif.0 ], [ %load1, %if.1 ]
  br i1 %cmp2, label %if.2, label %endif.2

if.2:
  %getElem2.0 = getelementptr i8, ptr %in, i64 %add
  %getElem2.1 = getelementptr inbounds i8, ptr %getElem2.0, i64 64
  %load2 = load i8, ptr %getElem2.1
  br label %endif.2

endif.2:
  %value2 = phi i8 [ 0, %endif.1 ], [ %load2, %if.2 ]
  %out0 = add i8 %value0, %value1
  %out1 = add i8 %out0, %value2
  store i8 %out1, ptr %out

  ret void
}

; CHECK-LABEL: psr_i8_base_delta(
; CHECK-SAME:      ptr [[IN:%.*]], ptr [[OUT:%.*]], i64 [[ADD:%.*]], i1 [[CMP0:%.*]], i1 [[CMP1:%.*]], i1 [[CMP2:%.*]])
; CHECK-NEXT: [[ENTRY:.*]]:
; CHECK-NEXT:   [[GEP:%.*]] = getelementptr i8, ptr [[IN]], i64 [[ADD]]
; CHECK-NEXT:   br i1 [[CMP0]], label %[[IF0:.*]], label %[[ENDIF0:.*]]
; CHECK-EMPTY:
; CHECK-NEXT: [[IF0]]:
; CHECK-NEXT:   [[GEP0:%.*]] = getelementptr inbounds i8, ptr [[GEP]], i64 32
; CHECK-NEXT:   [[LOAD0:%.*]] = load i8, ptr [[GEP0]]
; CHECK-NEXT:   br label %[[ENDIF0]]
; CHECK-EMPTY:
; CHECK-NEXT: [[ENDIF0]]:
; CHECK-NEXT:   [[VAL0:%.*]] = phi i8 [ 0, %[[ENTRY]] ], [ [[LOAD0]], %[[IF0]] ]
; CHECK-NEXT:   br i1 [[CMP1]], label %[[IF1:.*]], label %[[ENDIF1:.*]]
; CHECK-EMPTY:
; CHECK-NEXT: [[IF1]]:
; CHECK-NEXT:   [[GEP1_0:%.*]] = getelementptr inbounds i8, ptr [[GEP]], i64 1
; CHECK-NEXT:   [[GEP1_1:%.*]] = getelementptr inbounds i8, ptr [[GEP1_0]], i64 64
; CHECK-NEXT:   [[LOAD1:%.*]] = load i8, ptr [[GEP1_1]]
; CHECK-NEXT:   br label %[[ENDIF1]]
; CHECK-EMPTY:
; CHECK-NEXT: [[ENDIF1]]:
; CHECK-NEXT:   [[VAL1:%.*]] = phi i8 [ 0, %[[ENDIF0]] ], [ [[LOAD1]], %[[IF1]] ]
; CHECK-NEXT:   br i1 [[CMP2]], label %[[IF2:.*]], label %[[ENDIF2:.*]]
; CHECK-EMPTY:
; CHECK-NEXT: [[IF2]]:
; CHECK-NEXT:   [[GEP2_0:%.*]] = getelementptr inbounds i8, ptr [[GEP]], i64 2
; CHECK-NEXT:   [[GEP2_1:%.*]] = getelementptr inbounds i8, ptr [[GEP2_0]], i64 96
; CHECK-NEXT:   [[LOAD2:%.*]] = load i8, ptr [[GEP2_1]]
; CHECK-NEXT:   br label %[[ENDIF2]]
; CHECK-EMPTY:
; CHECK-NEXT: [[ENDIF2]]:
define void @psr_i8_base_delta(ptr %in, ptr %out, i64 %add, i1 %cmp0, i1 %cmp1, i1 %cmp2) {
entry:
  br i1 %cmp0, label %if.0, label %endif.0

if.0:
  %getElem0.0 = getelementptr inbounds i8, ptr %in, i64 %add
  %getElem0.1 = getelementptr inbounds i8, ptr %getElem0.0, i64 32
  %load0 = load i8, ptr %getElem0.1
  br label %endif.0

endif.0:
  %value0 = phi i8 [ 0, %entry ], [ %load0, %if.0 ]
  br i1 %cmp1, label %if.1, label %endif.1

if.1:
  %getElem1.0 = getelementptr inbounds i8, ptr %in, i64 1
  %getElem1.1 = getelementptr inbounds i8, ptr %getElem1.0, i64 %add
  %getElem1.2 = getelementptr inbounds i8, ptr %getElem1.1, i64 64
  %load1 = load i8, ptr %getElem1.2
  br label %endif.1

endif.1:
  %value1 = phi i8 [ 0, %endif.0 ], [ %load1, %if.1 ]
  br i1 %cmp2, label %if.2, label %endif.2

if.2:
  %getElem2.0 = getelementptr inbounds i8, ptr %in, i64 2
  %getElem2.1 = getelementptr inbounds i8, ptr %getElem2.0, i64 %add
  %getElem2.2 = getelementptr inbounds i8, ptr %getElem2.1, i64 96
  %load2 = load i8, ptr %getElem2.2
  br label %endif.2

endif.2:
  %value2 = phi i8 [ 0, %endif.1 ], [ %load2, %if.2 ]
  %out0 = add i8 %value0, %value1
  %out1 = add i8 %out0, %value2
  store i8 %out1, ptr %out

  ret void
}

; CHECK-LABEL: psr_i8_index_delta(
; CHECK-SAME:      ptr [[IN:%.*]], ptr [[OUT:%.*]], i64 [[ADD:%.*]], i1 [[CMP0:%.*]], i1 [[CMP1:%.*]], i1 [[CMP2:%.*]])
; CHECK-NEXT: [[ENTRY:.*]]:
; CHECK-NEXT:   br i1 [[CMP0]], label %[[IF0:.*]], label %[[ENDIF0:.*]]
; CHECK-EMPTY:
; CHECK-NEXT: [[IF0]]:
; CHECK-NEXT:   [[GEP:%.*]] = getelementptr inbounds double, ptr [[IN]], i64 [[ADD]]
; CHECK-NEXT:   [[GEP0:%.*]] = getelementptr inbounds i8, ptr [[GEP]], i64 32
; CHECK-NEXT:   [[LOAD0:%.*]] = load i8, ptr [[GEP0]]
; CHECK-NEXT:   br label %[[ENDIF0]]
; CHECK-EMPTY:
; CHECK-NEXT: [[ENDIF0]]:
; CHECK-NEXT:   [[VAL0:%.*]] = phi i8 [ 0, %[[ENTRY]] ], [ [[LOAD0]], %[[IF0]] ]
; CHECK-NEXT:   [[GEP1_0:%.*]] = getelementptr i8, ptr [[IN]], i64 [[ADD]]
; CHECK-NEXT:   br i1 [[CMP1]], label %[[IF1:.*]], label %[[ENDIF1:.*]]
; CHECK-EMPTY:
; CHECK-NEXT: [[IF1]]:
; CHECK-NEXT:   [[GEP1_1:%.*]] = getelementptr inbounds i8, ptr [[GEP1_0]], i64 64
; CHECK-NEXT:   [[LOAD1:%.*]] = load i8, ptr [[GEP1_1]]
; CHECK-NEXT:   br label %[[ENDIF1]]
; CHECK-EMPTY:
; CHECK-NEXT: [[ENDIF1]]:
; CHECK-NEXT:   [[VAL1:%.*]] = phi i8 [ 0, %[[ENDIF0]] ], [ [[LOAD1]], %[[IF1]] ]
; CHECK-NEXT:   br i1 [[CMP2]], label %[[IF2:.*]], label %[[ENDIF2:.*]]
; CHECK-EMPTY:
; CHECK-NEXT: [[IF2]]:
; CHECK-NEXT:   [[GEP2_1:%.*]] = getelementptr inbounds i8, ptr [[GEP1_0]], i64 96
; CHECK-NEXT:   [[LOAD2:%.*]] = load i8, ptr [[GEP2_1]]
; CHECK-NEXT:   br label %[[ENDIF2]]
; CHECK-EMPTY:
; CHECK-NEXT: [[ENDIF2]]:
define void @psr_i8_index_delta(ptr %in, ptr %out, i64 %add, i1 %cmp0, i1 %cmp1, i1 %cmp2) {
entry:
  br i1 %cmp0, label %if.0, label %endif.0

if.0:
  %getElem0.0 = getelementptr inbounds double, ptr %in, i64 %add
  %getElem0.1 = getelementptr inbounds i8, ptr %getElem0.0, i64 32
  %load0 = load i8, ptr %getElem0.1
  br label %endif.0

endif.0:
  %value0 = phi i8 [ 0, %entry ], [ %load0, %if.0 ]
  br i1 %cmp1, label %if.1, label %endif.1

if.1:
  %getElem1.0 = getelementptr inbounds i8, ptr %in, i64 %add
  %getElem1.1 = getelementptr inbounds i8, ptr %getElem1.0, i64 64
  %load1 = load i8, ptr %getElem1.1
  br label %endif.1

endif.1:
  %value1 = phi i8 [ 0, %endif.0 ], [ %load1, %if.1 ]
  br i1 %cmp2, label %if.2, label %endif.2

if.2:
  %getElem2.0 = getelementptr inbounds i8, ptr %in, i64 %add
  %getElem2.1 = getelementptr inbounds i8, ptr %getElem2.0, i64 96
  %load2 = load i8, ptr %getElem2.1
  br label %endif.2

endif.2:
  %value2 = phi i8 [ 0, %endif.1 ], [ %load2, %if.2 ]
  %out0 = add i8 %value0, %value1
  %out1 = add i8 %out0, %value2
  store i8 %out1, ptr %out

  ret void
}

; CHECK-LABEL: psr_i8_stride_delta(
; CHECK-SAME:      ptr [[IN:%.*]], ptr [[OUT:%.*]], i64 [[ADD:%.*]], i64 [[OFFSET:%.*]], i1 [[CMP0:%.*]], i1 [[CMP1:%.*]], i1 [[CMP2:%.*]])
; CHECK-NEXT: [[ENTRY:.*]]:
; CHECK-NEXT:   [[GEP:%.*]] = getelementptr i8, ptr [[IN]], i64 [[ADD]]
; CHECK-NEXT:   br i1 [[CMP0]], label %[[IF0:.*]], label %[[ENDIF0:.*]]
; CHECK-EMPTY:
; CHECK-NEXT: [[IF0]]:
; CHECK-NEXT:   [[GEP0:%.*]] = getelementptr inbounds i8, ptr [[GEP]], i64 32
; CHECK-NEXT:   [[LOAD0:%.*]] = load i8, ptr [[GEP0]]
; CHECK-NEXT:   br label %[[ENDIF0]]
; CHECK-EMPTY:
; CHECK-NEXT: [[ENDIF0]]:
; CHECK-NEXT:   [[VAL0:%.*]] = phi i8 [ 0, %[[ENTRY]] ], [ [[LOAD0]], %[[IF0]] ]
; CHECK-NEXT:   br i1 [[CMP1]], label %[[IF1:.*]], label %[[ENDIF1:.*]]
; CHECK-EMPTY:
; CHECK-NEXT: [[IF1]]:
; CHECK-NEXT:   [[GEP1_0:%.*]] = getelementptr inbounds i8, ptr [[GEP]], i64 1
; CHECK-NEXT:   [[GEP1_1:%.*]] = getelementptr inbounds i8, ptr [[GEP1_0]], i64 64
; CHECK-NEXT:   [[LOAD1:%.*]] = load i8, ptr [[GEP1_1]]
; CHECK-NEXT:   br label %[[ENDIF1]]
; CHECK-EMPTY:
; CHECK-NEXT: [[ENDIF1]]:
; CHECK-NEXT:   [[VAL1:%.*]] = phi i8 [ 0, %[[ENDIF0]] ], [ [[LOAD1]], %[[IF1]] ]
; CHECK-NEXT:   br i1 [[CMP2]], label %[[IF2:.*]], label %[[ENDIF2:.*]]
; CHECK-EMPTY:
; CHECK-NEXT: [[IF2]]:
; CHECK-NEXT:   [[ADD2:%.*]] = add i64 [[ADD]], [[OFFSET]]
; CHECK-NEXT:   [[GEP2_0:%.*]] = getelementptr inbounds i8, ptr [[IN]], i64 [[ADD2]]
; CHECK-NEXT:   [[GEP2_1:%.*]] = getelementptr inbounds i8, ptr [[GEP2_0]], i64 96
; CHECK-NEXT:   [[LOAD2:%.*]] = load i8, ptr [[GEP2_1]]
; CHECK-NEXT:   br label %[[ENDIF2]]
; CHECK-EMPTY:
; CHECK-NEXT: [[ENDIF2]]:
define void @psr_i8_stride_delta(ptr %in, ptr %out, i64 %add, i64 %offset, i1 %cmp0, i1 %cmp1, i1 %cmp2) {
entry:
  br i1 %cmp0, label %if.0, label %endif.0

if.0:
  %getElem0.0 = getelementptr inbounds i8, ptr %in, i64 %add
  %getElem0.1 = getelementptr inbounds i8, ptr %getElem0.0, i64 32
  %load0 = load i8, ptr %getElem0.1
  br label %endif.0

endif.0:
  %value0 = phi i8 [ 0, %entry ], [ %load0, %if.0 ]
  br i1 %cmp1, label %if.1, label %endif.1

if.1:
  %add1 = add i64 %add, 1
  %getElem1.0 = getelementptr inbounds i8, ptr %in, i64 %add1
  %getElem1.1 = getelementptr inbounds i8, ptr %getElem1.0, i64 64
  %load1 = load i8, ptr %getElem1.1
  br label %endif.1

endif.1:
  %value1 = phi i8 [ 0, %endif.0 ], [ %load1, %if.1 ]
  br i1 %cmp2, label %if.2, label %endif.2

if.2:
  %add2 = add i64 %add, %offset
  %getElem2.0 = getelementptr inbounds i8, ptr %in, i64 %add2
  %getElem2.1 = getelementptr inbounds i8, ptr %getElem2.0, i64 96
  %load2 = load i8, ptr %getElem2.1
  br label %endif.2

endif.2:
  %value2 = phi i8 [ 0, %endif.1 ], [ %load2, %if.2 ]
  %out0 = add i8 %value0, %value1
  %out1 = add i8 %out0, %value2
  store i8 %out1, ptr %out

  ret void
}
