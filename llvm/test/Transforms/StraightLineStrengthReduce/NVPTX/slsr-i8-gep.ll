; RUN: opt < %s -passes=slsr -S | FileCheck %s

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-unknown-unknown"

; CHECK-LABEL: slsr_i8_zero_delta(
; CHECK-SAME:      ptr [[IN:%.*]], ptr [[OUT:%.*]], i64 [[ADD:%.*]])
; CHECK-NEXT:   [[GEP:%.*]] = getelementptr inbounds i8, ptr [[IN]], i64 [[ADD]]
; CHECK-NEXT:   [[GEP0:%.*]] = getelementptr inbounds i8, ptr [[GEP]], i64 32
; CHECK-NEXT:   [[LOAD0:%.*]] = load i8, ptr [[GEP0]]
; CHECK-NEXT:   [[GEP1:%.*]] = getelementptr inbounds i8, ptr [[GEP]], i64 64
; CHECK-NEXT:   [[LOAD1:%.*]] = load i8, ptr [[GEP1]]
; CHECK-NEXT:   [[GEP2:%.*]] = getelementptr inbounds i8, ptr [[GEP]], i64 96
; CHECK-NEXT:   [[LOAD2:%.*]] = load i8, ptr [[GEP2]]
define void @slsr_i8_zero_delta(ptr %in, ptr %out, i64 %add) {
  %getElem0.0 = getelementptr inbounds i8, ptr %in, i64 %add
  %getElem0.1 = getelementptr inbounds i8, ptr %getElem0.0, i64 32
  %load0 = load i8, ptr %getElem0.1

  %getElem1.0 = getelementptr inbounds i8, ptr %in, i64 %add
  %getElem1.1 = getelementptr inbounds i8, ptr %getElem1.0, i64 64
  %load1 = load i8, ptr %getElem1.1

  %getElem2.0 = getelementptr inbounds i8, ptr %in, i64 %add
  %getElem2.1 = getelementptr inbounds i8, ptr %getElem2.0, i64 96
  %load2 = load i8, ptr %getElem2.1

  %out0 = add i8 %load0, %load1
  %out1 = add i8 %out0, %load2
  store i8 %out1, ptr %out

  ret void
}

; CHECK-LABEL: slsr_i8_zero_delta_2(
; CHECK-SAME:      ptr [[IN:%.*]], ptr [[OUT:%.*]], i64 [[ADD:%.*]])
; CHECK-NEXT:   [[GEP0:%.*]] = getelementptr inbounds i8, ptr [[IN]], i64 [[ADD]]
; CHECK-NEXT:   [[LOAD0:%.*]] = load i8, ptr [[GEP0]]
; CHECK-NEXT:   [[GEP1:%.*]] = getelementptr inbounds i8, ptr [[GEP0]], i64 32
; CHECK-NEXT:   [[LOAD1:%.*]] = load i8, ptr [[GEP1]]
; CHECK-NEXT:   [[GEP2:%.*]] = getelementptr inbounds i8, ptr [[GEP0]], i64 64
; CHECK-NEXT:   [[LOAD2:%.*]] = load i8, ptr [[GEP2]]
define void @slsr_i8_zero_delta_2(ptr %in, ptr %out, i64 %add) {
  %getElem0.0 = getelementptr inbounds i8, ptr %in, i64 %add
  %load0 = load i8, ptr %getElem0.0

  %getElem1.0 = getelementptr i8, ptr %in, i64 %add
  %getElem1.1 = getelementptr inbounds i8, ptr %getElem1.0, i64 32
  %load1 = load i8, ptr %getElem1.1

  %getElem2.0 = getelementptr i8, ptr %in, i64 %add
  %getElem2.1 = getelementptr inbounds i8, ptr %getElem2.0, i64 64
  %load2 = load i8, ptr %getElem2.1

  %out0 = add i8 %load0, %load1
  %out1 = add i8 %out0, %load2
  store i8 %out1, ptr %out

  ret void
}

; CHECK-LABEL: slsr_i8_base_delta(
; CHECK-SAME:      ptr [[IN:%.*]], ptr [[OUT:%.*]], i64 [[ADD:%.*]])
; CHECK-NEXT:   [[GEP:%.*]] = getelementptr inbounds i8, ptr [[IN]], i64 [[ADD]]
; CHECK-NEXT:   [[GEP0:%.*]] = getelementptr inbounds i8, ptr [[GEP]], i64 32
; CHECK-NEXT:   [[LOAD0:%.*]] = load i8, ptr [[GEP0]]
; CHECK-NEXT:   [[GEP1_0:%.*]] = getelementptr inbounds i8, ptr [[GEP]], i64 1
; CHECK-NEXT:   [[GEP1_1:%.*]] = getelementptr inbounds i8, ptr [[GEP1_0]], i64 64
; CHECK-NEXT:   [[LOAD1:%.*]] = load i8, ptr [[GEP1_1]]
; CHECK-NEXT:   [[GEP2_0:%.*]] = getelementptr inbounds i8, ptr [[GEP]], i64 2
; CHECK-NEXT:   [[GEP2_1:%.*]] = getelementptr inbounds i8, ptr [[GEP2_0]], i64 96
; CHECK-NEXT:   [[LOAD2:%.*]] = load i8, ptr [[GEP2_1]]
define void @slsr_i8_base_delta(ptr %in, ptr %out, i64 %add) {
  %getElem0.0 = getelementptr inbounds i8, ptr %in, i64 %add
  %getElem0.1 = getelementptr inbounds i8, ptr %getElem0.0, i64 32
  %load0 = load i8, ptr %getElem0.1

  %getElem1.0 = getelementptr inbounds i8, ptr %in, i64 1
  %getElem1.1 = getelementptr inbounds i8, ptr %getElem1.0, i64 %add
  %getElem1.2 = getelementptr inbounds i8, ptr %getElem1.1, i64 64
  %load1 = load i8, ptr %getElem1.2

  %getElem2.0 = getelementptr inbounds i8, ptr %in, i64 2
  %getElem2.1 = getelementptr inbounds i8, ptr %getElem2.0, i64 %add
  %getElem2.2 = getelementptr inbounds i8, ptr %getElem2.1, i64 96
  %load2 = load i8, ptr %getElem2.2

  %out0 = add i8 %load0, %load1
  %out1 = add i8 %out0, %load2
  store i8 %out1, ptr %out

  ret void
}

; CHECK-LABEL: slsr_i8_index_delta(
; CHECK-SAME:      ptr [[IN:%.*]], ptr [[OUT:%.*]], i64 [[ADD:%.*]])
; CHECK-NEXT:   [[GEP:%.*]] = getelementptr inbounds double, ptr [[IN]], i64 [[ADD]]
; CHECK-NEXT:   [[GEP0:%.*]] = getelementptr inbounds i8, ptr [[GEP]], i64 32
; CHECK-NEXT:   [[LOAD0:%.*]] = load i8, ptr [[GEP0]]
; CHECK-NEXT:   [[GEP1_0:%.*]] = getelementptr inbounds i8, ptr [[IN]], i64 [[ADD]]
; CHECK-NEXT:   [[GEP1_1:%.*]] = getelementptr inbounds i8, ptr [[GEP1_0]], i64 64
; CHECK-NEXT:   [[LOAD1:%.*]] = load i8, ptr [[GEP1_1]]
; CHECK-NEXT:   [[GEP2:%.*]] = getelementptr inbounds i8, ptr [[GEP1_0]], i64 96
; CHECK-NEXT:   [[LOAD2:%.*]] = load i8, ptr [[GEP2]]
define void @slsr_i8_index_delta(ptr %in, ptr %out, i64 %add) {
  %getElem0.0 = getelementptr inbounds double, ptr %in, i64 %add
  %getElem0.1 = getelementptr inbounds i8, ptr %getElem0.0, i64 32
  %load0 = load i8, ptr %getElem0.1

  %getElem1.0 = getelementptr inbounds i8, ptr %in, i64 %add
  %getElem1.1 = getelementptr inbounds i8, ptr %getElem1.0, i64 64
  %load1 = load i8, ptr %getElem1.1

  %getElem2.0 = getelementptr inbounds i8, ptr %in, i64 %add
  %getElem2.1 = getelementptr inbounds i8, ptr %getElem2.0, i64 96
  %load2 = load i8, ptr %getElem2.1

  %out0 = add i8 %load0, %load1
  %out1 = add i8 %out0, %load2
  store i8 %out1, ptr %out

  ret void
}

; CHECK-LABEL: slsr_i8_stride_delta(
; CHECK-SAME:      ptr [[IN:%.*]], ptr [[OUT:%.*]], i64 [[ADD:%.*]], i64 [[OFFSET:%.*]])
; CHECK-NEXT:   [[GEP:%.*]] = getelementptr inbounds i8, ptr [[IN]], i64 [[ADD]]
; CHECK-NEXT:   [[GEP0:%.*]] = getelementptr inbounds i8, ptr [[GEP]], i64 32
; CHECK-NEXT:   [[LOAD0:%.*]] = load i8, ptr [[GEP0]]
; CHECK-NEXT:   [[GEP1_0:%.*]] = getelementptr inbounds i8, ptr [[GEP]], i64 1
; CHECK-NEXT:   [[GEP1_1:%.*]] = getelementptr inbounds i8, ptr [[GEP1_0]], i64 64
; CHECK-NEXT:   [[LOAD1:%.*]] = load i8, ptr [[GEP1_1]]
; CHECK-NEXT:   [[GEP2_0:%.*]] = getelementptr inbounds i8, ptr [[GEP]], i64 [[OFFSET]]
; CHECK-NEXT:   [[GEP2_1:%.*]] = getelementptr inbounds i8, ptr [[GEP2_0]], i64 96
; CHECK-NEXT:   [[LOAD2:%.*]] = load i8, ptr [[GEP2_1]]
define void @slsr_i8_stride_delta(ptr %in, ptr %out, i64 %add, i64 %offset) {
  %getElem0.0 = getelementptr inbounds i8, ptr %in, i64 %add
  %getElem0.1 = getelementptr inbounds i8, ptr %getElem0.0, i64 32
  %load0 = load i8, ptr %getElem0.1

  %add1 = add i64 %add, 1
  %getElem1.0 = getelementptr inbounds i8, ptr %in, i64 %add1
  %getElem1.1 = getelementptr inbounds i8, ptr %getElem1.0, i64 64
  %load1 = load i8, ptr %getElem1.1

  %add2 = add i64 %add, %offset
  %getElem2.0 = getelementptr inbounds i8, ptr %in, i64 %add2
  %getElem2.1 = getelementptr inbounds i8, ptr %getElem2.0, i64 96
  %load2 = load i8, ptr %getElem2.1

  %out0 = add i8 %load0, %load1
  %out1 = add i8 %out0, %load2
  store i8 %out1, ptr %out

  ret void
}
