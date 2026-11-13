; RUN: opt < %s -passes=slsr -enable-partial-strength-reduce -S | FileCheck %s
; RUN: opt < %s -passes=slsr -enable-partial-strength-reduce -S -max-psr-path-length=2 | FileCheck %s --check-prefix=PATH_LIMIT

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-unknown-unknown"

; CHECK-LABEL: psr_zero_delta(
; CHECK-SAME:      ptr [[IN:%.*]], ptr [[OUT:%.*]], i64 [[ADD:%.*]], i1 [[CMP0:%.*]], i1 [[CMP1:%.*]], i1 [[CMP2:%.*]])
; CHECK-NEXT: [[ENTRY:.*]]:
; CHECK-NEXT:   [[GEP:%.*]] = getelementptr float, ptr [[IN]], i64 [[ADD]]
; CHECK-NEXT:   br i1 [[CMP0]], label %[[IF0:.*]], label %[[ENDIF0:.*]]
; CHECK-EMPTY:
; CHECK-NEXT: [[IF0]]:
; CHECK-NEXT:   [[GEP0:%.*]] = getelementptr inbounds float, ptr [[GEP]], i64 32
; CHECK-NEXT:   [[LOAD0:%.*]] = load float, ptr [[GEP0]]
; CHECK-NEXT:   br label %[[ENDIF0]]
; CHECK-EMPTY:
; CHECK-NEXT: [[ENDIF0]]:
; CHECK-NEXT:   [[VAL0:%.*]] = phi float [ 0.000000e+00, %[[ENTRY]] ], [ [[LOAD0]], %[[IF0]] ]
; CHECK-NEXT:   br i1 [[CMP1]], label %[[IF1:.*]], label %[[ENDIF1:.*]]
; CHECK-EMPTY:
; CHECK-NEXT: [[IF1]]:
; CHECK-NEXT:   [[GEP1:%.*]] = getelementptr inbounds float, ptr [[GEP]], i64 64
; CHECK-NEXT:   [[LOAD1:%.*]] = load float, ptr [[GEP1]]
; CHECK-NEXT:   br label %[[ENDIF1]]
; CHECK-EMPTY:
; CHECK-NEXT: [[ENDIF1]]:
; CHECK-NEXT:   [[VAL1:%.*]] = phi float [ 0.000000e+00, %[[ENDIF0]] ], [ [[LOAD1]], %[[IF1]] ]
; CHECK-NEXT:   br i1 [[CMP2]], label %[[IF2:.*]], label %[[ENDIF2:.*]]
; CHECK-EMPTY:
; CHECK-NEXT: [[IF2]]:
; CHECK-NEXT:   [[GEP2:%.*]] = getelementptr inbounds float, ptr [[GEP]], i64 96
; CHECK-NEXT:   [[LOAD2:%.*]] = load float, ptr [[GEP2]]
; CHECK-NEXT:   br label %[[ENDIF2]]
; CHECK-EMPTY:
; CHECK-NEXT: [[ENDIF2]]:
define void @psr_zero_delta(ptr %in, ptr %out, i64 %add, i1 %cmp.0, i1 %cmp1, i1 %cmp2) {
entry:
  br i1 %cmp.0, label %if.0, label %endif.0

if.0:
  %getElem0.0 = getelementptr inbounds float, ptr %in, i64 %add
  %getElem0.1 = getelementptr inbounds float, ptr %getElem0.0, i64 32
  %load0 = load float, ptr %getElem0.1
  br label %endif.0

endif.0:
  %value0 = phi float [ 0.000000e+00, %entry ], [ %load0, %if.0 ]
  br i1 %cmp1, label %if.1, label %endif.1

if.1:
  %getElem1.0 = getelementptr inbounds float, ptr %in, i64 %add
  %getElem1.1 = getelementptr inbounds float, ptr %getElem1.0, i64 64
  %load1 = load float, ptr %getElem1.1
  br label %endif.1

endif.1:
  %value1 = phi float [ 0.000000e+00, %endif.0 ], [ %load1, %if.1 ]
  br i1 %cmp2, label %if.2, label %endif.2

if.2:
  %getElem2.0 = getelementptr inbounds float, ptr %in, i64 %add
  %getElem2.1 = getelementptr inbounds float, ptr %getElem2.0, i64 96
  %load2 = load float, ptr %getElem2.1
  br label %endif.2

endif.2:
  %value2 = phi float [ 0.000000e+00, %endif.1 ], [ %load2, %if.2 ]
  %out0 = fadd float %value0, %value1
  %out1 = fadd float %out0, %value2
  store float %out1, ptr %out

  ret void
}

; CHECK-LABEL: psr_zero_delta_2(
; CHECK-SAME:      ptr [[IN:%.*]], ptr [[OUT:%.*]], i64 [[ADD:%.*]], i1 [[CMP0:%.*]], i1 [[CMP1:%.*]], i1 [[CMP2:%.*]])
; CHECK-NEXT: [[ENTRY:.*]]:
; CHECK-NEXT:   [[GEP0:%.*]] = getelementptr float, ptr [[IN]], i64 [[ADD]]
; CHECK-NEXT:   br i1 [[CMP0]], label %[[IF0:.*]], label %[[ENDIF0:.*]]
; CHECK-EMPTY:
; CHECK-NEXT: [[IF0]]:
; CHECK-NEXT:   [[LOAD0:%.*]] = load float, ptr [[GEP0]]
; CHECK-NEXT:   br label %[[ENDIF0]]
; CHECK-EMPTY:
; CHECK-NEXT: [[ENDIF0]]:
; CHECK-NEXT:   [[VAL0:%.*]] = phi float [ 0.000000e+00, %[[ENTRY]] ], [ [[LOAD0]], %[[IF0]] ]
; CHECK-NEXT:   br i1 [[CMP1]], label %[[IF1:.*]], label %[[ENDIF1:.*]]
; CHECK-EMPTY:
; CHECK-NEXT: [[IF1]]:
; CHECK-NEXT:   [[GEP1:%.*]] = getelementptr inbounds float, ptr [[GEP0]], i64 32
; CHECK-NEXT:   [[LOAD1:%.*]] = load float, ptr [[GEP1]]
; CHECK-NEXT:   br label %[[ENDIF1]]
; CHECK-EMPTY:
; CHECK-NEXT: [[ENDIF1]]:
; CHECK-NEXT:   [[VAL1:%.*]] = phi float [ 0.000000e+00, %[[ENDIF0]] ], [ [[LOAD1]], %[[IF1]] ]
; CHECK-NEXT:   br i1 [[CMP2]], label %[[IF2:.*]], label %[[ENDIF2:.*]]
; CHECK-EMPTY:
; CHECK-NEXT: [[IF2]]:
; CHECK-NEXT:   [[GEP2:%.*]] = getelementptr inbounds float, ptr [[GEP0]], i64 64
; CHECK-NEXT:   [[LOAD2:%.*]] = load float, ptr [[GEP2]]
; CHECK-NEXT:   br label %[[ENDIF2]]
; CHECK-EMPTY:
; CHECK-NEXT: [[ENDIF2]]:

define void @psr_zero_delta_2(ptr %in, ptr %out, i64 %add, i1 %cmp.0, i1 %cmp1, i1 %cmp2) {
entry:
  br i1 %cmp.0, label %if.0, label %endif.0

if.0:
  %getElem0.0 = getelementptr inbounds float, ptr %in, i64 %add
  %load0 = load float, ptr %getElem0.0
  br label %endif.0

endif.0:
  %value0 = phi float [ 0.000000e+00, %entry ], [ %load0, %if.0 ]
  br i1 %cmp1, label %if.1, label %endif.1

if.1:
  %getElem1.0 = getelementptr float, ptr %in, i64 %add
  %getElem1.1 = getelementptr inbounds float, ptr %getElem1.0, i64 32
  %load1 = load float, ptr %getElem1.1
  br label %endif.1

endif.1:
  %value1 = phi float [ 0.000000e+00, %endif.0 ], [ %load1, %if.1 ]
  br i1 %cmp2, label %if.2, label %endif.2

if.2:
  %getElem2.0 = getelementptr float, ptr %in, i64 %add
  %getElem2.1 = getelementptr inbounds float, ptr %getElem2.0, i64 64
  %load2 = load float, ptr %getElem2.1
  br label %endif.2

endif.2:
  %value2 = phi float [ 0.000000e+00, %endif.1 ], [ %load2, %if.2 ]
  %out0 = fadd float %value0, %value1
  %out1 = fadd float %out0, %value2
  store float %out1, ptr %out

  ret void
}

; CHECK-LABEL: psr_base_delta(
; CHECK-SAME:      ptr [[IN:%.*]], ptr [[OUT:%.*]], i64 [[ADD:%.*]], i1 [[CMP0:%.*]], i1 [[CMP1:%.*]], i1 [[CMP2:%.*]])
; CHECK-NEXT: [[ENTRY:.*]]:
; CHECK-NEXT:   [[GEP:%.*]] = getelementptr float, ptr [[IN]], i64 [[ADD]]
; CHECK-NEXT:   br i1 [[CMP0]], label %[[IF0:.*]], label %[[ENDIF0:.*]]
; CHECK-EMPTY:
; CHECK-NEXT: [[IF0]]:
; CHECK-NEXT:   [[GEP0:%.*]] = getelementptr inbounds float, ptr [[GEP]], i64 32
; CHECK-NEXT:   [[LOAD0:%.*]] = load float, ptr [[GEP0]]
; CHECK-NEXT:   br label %[[ENDIF0]]
; CHECK-EMPTY:
; CHECK-NEXT: [[ENDIF0]]:
; CHECK-NEXT:   [[VAL0:%.*]] = phi float [ 0.000000e+00, %[[ENTRY]] ], [ [[LOAD0]], %[[IF0]] ]
; CHECK-NEXT:   br i1 [[CMP1]], label %[[IF1:.*]], label %[[ENDIF1:.*]]
; CHECK-EMPTY:
; CHECK-NEXT: [[IF1]]:
; CHECK-NEXT:   [[GEP1_0:%.*]] = getelementptr inbounds i8, ptr [[GEP]], i64 4
; CHECK-NEXT:   [[GEP1_1:%.*]] = getelementptr inbounds float, ptr [[GEP1_0]], i64 64
; CHECK-NEXT:   [[LOAD1:%.*]] = load float, ptr [[GEP1_1]]
; CHECK-NEXT:   br label %[[ENDIF1]]
; CHECK-EMPTY:
; CHECK-NEXT: [[ENDIF1]]:
; CHECK-NEXT:   [[VAL1:%.*]] = phi float [ 0.000000e+00, %[[ENDIF0]] ], [ [[LOAD1]], %[[IF1]] ]
; CHECK-NEXT:   br i1 [[CMP2]], label %[[IF2:.*]], label %[[ENDIF2:.*]]
; CHECK-EMPTY:
; CHECK-NEXT: [[IF2]]:
; CHECK-NEXT:   [[GEP2_0:%.*]] = getelementptr inbounds i8, ptr [[GEP]], i64 8
; CHECK-NEXT:   [[GEP2_1:%.*]] = getelementptr inbounds float, ptr [[GEP2_0]], i64 96
; CHECK-NEXT:   [[LOAD2:%.*]] = load float, ptr [[GEP2_1]]
; CHECK-NEXT:   br label %[[ENDIF2]]
; CHECK-EMPTY:
; CHECK-NEXT: [[ENDIF2]]:

; PATH_LIMIT-LABEL: psr_base_delta(
; PATH_LIMIT-SAME:      ptr [[IN:%.*]], ptr [[OUT:%.*]], i64 [[ADD:%.*]], i1 [[CMP0:%.*]], i1 [[CMP1:%.*]], i1 [[CMP2:%.*]])
; PATH_LIMIT-NEXT: [[ENTRY:.*]]:
; PATH_LIMIT-NEXT:   [[GEP:%.*]] = getelementptr float, ptr [[IN]], i64 [[ADD]]
; PATH_LIMIT-NEXT:   br i1 [[CMP0]], label %[[IF0:.*]], label %[[ENDIF0:.*]]
; PATH_LIMIT-EMPTY:
; PATH_LIMIT-NEXT: [[IF0]]:
; PATH_LIMIT-NEXT:   [[GEP0:%.*]] = getelementptr inbounds float, ptr [[GEP]], i64 32
; PATH_LIMIT-NEXT:   [[LOAD0:%.*]] = load float, ptr [[GEP0]]
; PATH_LIMIT-NEXT:   br label %[[ENDIF0]]
; PATH_LIMIT-EMPTY:
; PATH_LIMIT-NEXT: [[ENDIF0]]:
; PATH_LIMIT-NEXT:   [[VAL0:%.*]] = phi float [ 0.000000e+00, %[[ENTRY]] ], [ [[LOAD0]], %[[IF0]] ]
; PATH_LIMIT-NEXT:   [[GEP1_0:%.*]] = getelementptr i8, ptr [[GEP]], i64 4
; PATH_LIMIT-NEXT:   br i1 [[CMP1]], label %[[IF1:.*]], label %[[ENDIF1:.*]]
; PATH_LIMIT-EMPTY:
; PATH_LIMIT-NEXT: [[IF1]]:
; PATH_LIMIT-NEXT:   [[GEP1_1:%.*]] = getelementptr inbounds float, ptr [[GEP1_0]], i64 64
; PATH_LIMIT-NEXT:   [[LOAD1:%.*]] = load float, ptr [[GEP1_1]]
; PATH_LIMIT-NEXT:   br label %[[ENDIF1]]
; PATH_LIMIT-EMPTY:
; PATH_LIMIT-NEXT: [[ENDIF1]]:
; PATH_LIMIT-NEXT:   [[VAL1:%.*]] = phi float [ 0.000000e+00, %[[ENDIF0]] ], [ [[LOAD1]], %[[IF1]] ]
; PATH_LIMIT-NEXT:   br i1 [[CMP2]], label %[[IF2:.*]], label %[[ENDIF2:.*]]
; PATH_LIMIT-EMPTY:
; PATH_LIMIT-NEXT: [[IF2]]:
; PATH_LIMIT-NEXT:   [[GEP2_0:%.*]] = getelementptr inbounds i8, ptr [[GEP1_0]], i64 4
; PATH_LIMIT-NEXT:   [[GEP2_1:%.*]] = getelementptr inbounds float, ptr [[GEP2_0]], i64 96
; PATH_LIMIT-NEXT:   [[LOAD2:%.*]] = load float, ptr [[GEP2_1]]
; PATH_LIMIT-NEXT:   br label %[[ENDIF2]]
; PATH_LIMIT-EMPTY:
; PATH_LIMIT-NEXT: [[ENDIF2]]:

define void @psr_base_delta(ptr %in, ptr %out, i64 %add, i1 %cmp.0, i1 %cmp1, i1 %cmp2) {
entry:
  br i1 %cmp.0, label %if.0, label %endif.0

if.0:
  %getElem0.0 = getelementptr inbounds float, ptr %in, i64 %add
  %getElem0.1 = getelementptr inbounds float, ptr %getElem0.0, i64 32
  %load0 = load float, ptr %getElem0.1
  br label %endif.0

endif.0:
  %value0 = phi float [ 0.000000e+00, %entry ], [ %load0, %if.0 ]
  br i1 %cmp1, label %if.1, label %endif.1

if.1:
  %getElem1.0 = getelementptr inbounds float, ptr %in, i64 1
  %getElem1.1 = getelementptr inbounds float, ptr %getElem1.0, i64 %add
  %getElem1.2 = getelementptr inbounds float, ptr %getElem1.1, i64 64
  %load1 = load float, ptr %getElem1.2
  br label %endif.1

endif.1:
  %value1 = phi float [ 0.000000e+00, %endif.0 ], [ %load1, %if.1 ]
  br i1 %cmp2, label %if.2, label %endif.2

if.2:
  %getElem2.0 = getelementptr inbounds float, ptr %in, i64 2
  %getElem2.1 = getelementptr inbounds float, ptr %getElem2.0, i64 %add
  %getElem2.2 = getelementptr inbounds float, ptr %getElem2.1, i64 96
  %load2 = load float, ptr %getElem2.2
  br label %endif.2

endif.2:
  %value2 = phi float [ 0.000000e+00, %endif.1 ], [ %load2, %if.2 ]
  %out0 = fadd float %value0, %value1
  %out1 = fadd float %out0, %value2
  store float %out1, ptr %out

  ret void
}

; CHECK-LABEL: psr_index_delta(
; CHECK-SAME:      ptr [[IN:%.*]], ptr [[OUT:%.*]], i64 [[ADD:%.*]], i1 [[CMP0:%.*]], i1 [[CMP1:%.*]], i1 [[CMP2:%.*]])
; CHECK-NEXT: [[ENTRY:.*]]:
; CHECK-NEXT:   [[GEP:%.*]] = getelementptr double, ptr [[IN]], i64 [[ADD]]
; CHECK-NEXT:   br i1 [[CMP0]], label %[[IF0:.*]], label %[[ENDIF0:.*]]
; CHECK-EMPTY:
; CHECK-NEXT: [[IF0]]:
; CHECK-NEXT:   [[GEP0:%.*]] = getelementptr inbounds float, ptr [[GEP]], i64 32
; CHECK-NEXT:   [[LOAD0:%.*]] = load float, ptr [[GEP0]]
; CHECK-NEXT:   br label %[[ENDIF0]]
; CHECK-EMPTY:
; CHECK-NEXT: [[ENDIF0]]:
; CHECK-NEXT:   [[VAL0:%.*]] = phi float [ 0.000000e+00, %[[ENTRY]] ], [ [[LOAD0]], %[[IF0]] ]
; CHECK-NEXT:   br i1 [[CMP1]], label %[[IF1:.*]], label %[[ENDIF1:.*]]
; CHECK-EMPTY:
; CHECK-NEXT: [[IF1]]:
; CHECK-NEXT:   [[SHL1:%.*]] = shl i64 [[ADD]], 2
; CHECK-NEXT:   [[SUB1:%.*]] = sub i64 0, [[SHL1]]
; CHECK-NEXT:   [[GEP1_0:%.*]] = getelementptr inbounds i8, ptr [[GEP]], i64 [[SUB1]]
; CHECK-NEXT:   [[GEP1_1:%.*]] = getelementptr inbounds float, ptr [[GEP1_0]], i64 64
; CHECK-NEXT:   [[LOAD1:%.*]] = load float, ptr [[GEP1_1]]
; CHECK-NEXT:   br label %[[ENDIF1]]
; CHECK-EMPTY:
; CHECK-NEXT: [[ENDIF1]]:
; CHECK-NEXT:   [[VAL1:%.*]] = phi float [ 0.000000e+00, %[[ENDIF0]] ], [ [[LOAD1]], %[[IF1]] ]
; CHECK-NEXT:   br i1 [[CMP2]], label %[[IF2:.*]], label %[[ENDIF2:.*]]
; CHECK-EMPTY:
; CHECK-NEXT: [[IF2]]:
; CHECK-NEXT:   [[SHL2:%.*]] = shl i64 [[ADD]], 2
; CHECK-NEXT:   [[SUB2:%.*]] = sub i64 0, [[SHL2]]
; CHECK-NEXT:   [[GEP2_0:%.*]] = getelementptr inbounds i8, ptr [[GEP]], i64 [[SUB2]]
; CHECK-NEXT:   [[GEP2_1:%.*]] = getelementptr inbounds float, ptr [[GEP2_0]], i64 96
; CHECK-NEXT:   [[LOAD2:%.*]] = load float, ptr [[GEP2_1]]
; CHECK-NEXT:   br label %[[ENDIF2]]
; CHECK-EMPTY:
; CHECK-NEXT: [[ENDIF2]]:
define void @psr_index_delta(ptr %in, ptr %out, i64 %add, i1 %cmp.0, i1 %cmp1, i1 %cmp2) {
entry:
  br i1 %cmp.0, label %if.0, label %endif.0

if.0:
  %getElem0.0 = getelementptr inbounds double, ptr %in, i64 %add
  %getElem0.1 = getelementptr inbounds float, ptr %getElem0.0, i64 32
  %load0 = load float, ptr %getElem0.1
  br label %endif.0

endif.0:
  %value0 = phi float [ 0.000000e+00, %entry ], [ %load0, %if.0 ]
  br i1 %cmp1, label %if.1, label %endif.1

if.1:
  %getElem1.0 = getelementptr inbounds float, ptr %in, i64 %add
  %getElem1.1 = getelementptr inbounds float, ptr %getElem1.0, i64 64
  %load1 = load float, ptr %getElem1.1
  br label %endif.1

endif.1:
  %value1 = phi float [ 0.000000e+00, %endif.0 ], [ %load1, %if.1 ]
  br i1 %cmp2, label %if.2, label %endif.2

if.2:
  %getElem2.0 = getelementptr inbounds float, ptr %in, i64 %add
  %getElem2.1 = getelementptr inbounds float, ptr %getElem2.0, i64 96
  %load2 = load float, ptr %getElem2.1
  br label %endif.2

endif.2:
  %value2 = phi float [ 0.000000e+00, %endif.1 ], [ %load2, %if.2 ]
  %out0 = fadd float %value0, %value1
  %out1 = fadd float %out0, %value2
  store float %out1, ptr %out

  ret void
}

; CHECK-LABEL: psr_stride_delta(
; CHECK-SAME:      ptr [[IN:%.*]], ptr [[OUT:%.*]], i64 [[ADD:%.*]], i64 [[OFFSET:%.*]], i1 [[CMP0:%.*]], i1 [[CMP1:%.*]], i1 [[CMP2:%.*]])
; CHECK-NEXT: [[ENTRY:.*]]:
; CHECK-NEXT:   [[GEP:%.*]] = getelementptr float, ptr [[IN]], i64 [[ADD]]
; CHECK-NEXT:   br i1 [[CMP0]], label %[[IF0:.*]], label %[[ENDIF0:.*]]
; CHECK-EMPTY:
; CHECK-NEXT: [[IF0]]:
; CHECK-NEXT:   [[GEP0:%.*]] = getelementptr inbounds float, ptr [[GEP]], i64 32
; CHECK-NEXT:   [[LOAD0:%.*]] = load float, ptr [[GEP0]]
; CHECK-NEXT:   br label %[[ENDIF0]]
; CHECK-EMPTY:
; CHECK-NEXT: [[ENDIF0]]:
; CHECK-NEXT:   [[VAL0:%.*]] = phi float [ 0.000000e+00, %[[ENTRY]] ], [ [[LOAD0]], %[[IF0]] ]
; CHECK-NEXT:   br i1 [[CMP1]], label %[[IF1:.*]], label %[[ENDIF1:.*]]
; CHECK-EMPTY:
; CHECK-NEXT: [[IF1]]:
; CHECK-NEXT:   [[GEP1_0:%.*]] = getelementptr inbounds i8, ptr [[GEP]], i64 4
; CHECK-NEXT:   [[GEP1_1:%.*]] = getelementptr inbounds float, ptr [[GEP1_0]], i64 64
; CHECK-NEXT:   [[LOAD1:%.*]] = load float, ptr [[GEP1_1]]
; CHECK-NEXT:   br label %[[ENDIF1]]
; CHECK-EMPTY:
; CHECK-NEXT: [[ENDIF1]]:
; CHECK-NEXT:   [[VAL1:%.*]] = phi float [ 0.000000e+00, %[[ENDIF0]] ], [ [[LOAD1]], %[[IF1]] ]
; CHECK-NEXT:   br i1 [[CMP2]], label %[[IF2:.*]], label %[[ENDIF2:.*]]
; CHECK-EMPTY:
; CHECK-NEXT: [[IF2]]:
; CHECK-NEXT:   [[ADD2:%.*]] = add i64 [[ADD]], [[OFFSET]]
; CHECK-NEXT:   [[GEP2_0:%.*]] = getelementptr inbounds float, ptr [[IN]], i64 [[ADD2]]
; CHECK-NEXT:   [[GEP2_1:%.*]] = getelementptr inbounds float, ptr [[GEP2_0]], i64 96
; CHECK-NEXT:   [[LOAD2:%.*]] = load float, ptr [[GEP2_1]]
; CHECK-NEXT:   br label %[[ENDIF2]]
; CHECK-EMPTY:
; CHECK-NEXT: [[ENDIF2]]:
define void @psr_stride_delta(ptr %in, ptr %out, i64 %add, i64 %offset, i1 %cmp.0, i1 %cmp1, i1 %cmp2) {
entry:
  br i1 %cmp.0, label %if.0, label %endif.0

if.0:
  %getElem0.0 = getelementptr inbounds float, ptr %in, i64 %add
  %getElem0.1 = getelementptr inbounds float, ptr %getElem0.0, i64 32
  %load0 = load float, ptr %getElem0.1
  br label %endif.0

endif.0:
  %value0 = phi float [ 0.000000e+00, %entry ], [ %load0, %if.0 ]
  br i1 %cmp1, label %if.1, label %endif.1

if.1:
  %add1 = add i64 %add, 1
  %getElem1.0 = getelementptr inbounds float, ptr %in, i64 %add1
  %getElem1.1 = getelementptr inbounds float, ptr %getElem1.0, i64 64
  %load1 = load float, ptr %getElem1.1
  br label %endif.1

endif.1:
  %value1 = phi float [ 0.000000e+00, %endif.0 ], [ %load1, %if.1 ]
  br i1 %cmp2, label %if.2, label %endif.2

if.2:
  %add2 = add i64 %add, %offset
  %getElem2.0 = getelementptr inbounds float, ptr %in, i64 %add2
  %getElem2.1 = getelementptr inbounds float, ptr %getElem2.0, i64 96
  %load2 = load float, ptr %getElem2.1
  br label %endif.2

endif.2:
  %value2 = phi float [ 0.000000e+00, %endif.1 ], [ %load2, %if.2 ]
  %out0 = fadd float %value0, %value1
  %out1 = fadd float %out0, %value2
  store float %out1, ptr %out

  ret void
}
