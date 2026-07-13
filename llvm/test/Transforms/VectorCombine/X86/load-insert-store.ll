; RUN: opt -S -passes=vector-combine -mtriple=x86_64-unknown-linux-gnu -mcpu=x86-64 < %s | FileCheck %s
; RUN: opt -S -passes=vector-combine -mtriple=x86_64-unknown-linux-gnu -mcpu=x86-64-v2 < %s | FileCheck %s
; RUN: opt -S -passes=vector-combine -mtriple=x86_64-unknown-linux-gnu -mcpu=x86-64-v3 < %s | FileCheck %s
; RUN: opt -S -passes=vector-combine -mtriple=x86_64-unknown-linux-gnu -mcpu=x86-64-v4 < %s | FileCheck %s

; Scalarization profitability is target-dependent. Check it for each x86-64
; microarchitecture level that this transform is expected to support.
define void @insert_store2(ptr %p, i16 %x, i16 %y) {
; CHECK-LABEL: @insert_store2(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[GEP0:%.*]] = getelementptr inbounds <8 x i16>, ptr [[P:%.*]], i32 0, i32 6
; CHECK-NEXT:    store i16 [[X:%.*]], ptr [[GEP0]], align 1
; CHECK-NEXT:    [[GEP1:%.*]] = getelementptr inbounds <8 x i16>, ptr [[P]], i32 0, i32 7
; CHECK-NEXT:    store i16 [[Y:%.*]], ptr [[GEP1]], align 1
; CHECK-NEXT:    ret void
;
entry:
  %load = load <8 x i16>, ptr %p, align 1
  %insert0 = insertelement <8 x i16> %load, i16 %x, i32 6
  %insert1 = insertelement <8 x i16> %insert0, i16 %y, i32 7
  store <8 x i16> %insert1, ptr %p, align 1
  ret void
}

; Keep the program order of stores when two inserts select the same element.
define void @insert_store_duplicate(ptr %p, i16 %x, i16 %y) {
; CHECK-LABEL: @insert_store_duplicate(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[GEP0:%.*]] = getelementptr inbounds <8 x i16>, ptr [[P:%.*]], i32 0, i32 3
; CHECK-NEXT:    store i16 [[X:%.*]], ptr [[GEP0]], align 1
; CHECK-NEXT:    [[GEP1:%.*]] = getelementptr inbounds <8 x i16>, ptr [[P]], i32 0, i32 3
; CHECK-NEXT:    store i16 [[Y:%.*]], ptr [[GEP1]], align 1
; CHECK-NEXT:    ret void
;
entry:
  %load = load <8 x i16>, ptr %p, align 1
  %insert0 = insertelement <8 x i16> %load, i16 %x, i32 3
  %insert1 = insertelement <8 x i16> %insert0, i16 %y, i32 3
  store <8 x i16> %insert1, ptr %p, align 1
  ret void
}
