; RUN: opt -passes=argpromotion -S %s | FileCheck %s

define internal i32 @callee2(i32* noundef %0) {
; CHECK-LABEL: define {{[^@]+}}@callee2
; CHECK-SAME: (i32 [[P_0:%.*]], i32 [[P_1:%.*]]) {
; CHECK-NEXT:    [[SUM:%.*]] = add nsw i32 [[P_0]], [[P_1]]
; CHECK-NEXT:    ret i32 [[SUM]]
;
  %2 = getelementptr inbounds i32, i32* %0, i64 0
  %3 = load i32, i32* %2, align 4
  %4 = getelementptr inbounds i32, i32* %0, i64 1
  %5 = load i32, i32* %4, align 4
  %6 = add nsw i32 %3, %5
  ret i32 %6
}

define i32 @caller2(i32 %0, i32 %1) {
; CHECK-LABEL: define {{[^@]+}}@caller2
; CHECK-SAME: (i32 [[P_0:%.*]], i32 [[P_1:%.*]]) {
; CHECK-NEXT:    [[TMP1:%.*]] = alloca [2 x i32], align 4
; CHECK-NEXT:    [[PL_0:%.*]] = getelementptr inbounds [2 x i32], [2 x i32]* [[TMP1]], i64 0, i64 0
; CHECK-NEXT:    store i32 [[P_0]], i32* [[PL_0]], align 4
; CHECK-NEXT:    [[PL_1:%.*]] = getelementptr inbounds i32, i32* [[PL_0]], i64 1
; CHECK-NEXT:    store i32 [[P_1]], i32* [[PL_1]], align 4
; CHECK-NEXT:    [[PL_2:%.*]] = getelementptr inbounds [2 x i32], [2 x i32]* [[TMP1]], i64 0, i64 0
; CHECK-NEXT:    [[VAL_0:%.*]] = load i32, i32* [[PL_2]], align 4
; CHECK-NEXT:    [[PL_3:%.*]] = getelementptr i32, i32* [[PL_2]], i64 1
; CHECK-NEXT:    [[VAL_1:%.*]] = load i32, i32* [[PL_3]], align 4
; CHECK-NEXT:    [[RES:%.*]] = call i32 @callee2(i32 [[VAL_0]], i32 [[VAL_1]])
; CHECK-NEXT:    ret i32 [[RES]]
;
  %3 = alloca [2 x i32], align 4
  %4 = getelementptr inbounds [2 x i32], [2 x i32]* %3, i64 0, i64 0
  store i32 %0, i32* %4, align 4
  %5 = getelementptr inbounds i32, i32* %4, i64 1
  store i32 %1, i32* %5, align 4
  %6 = getelementptr inbounds [2 x i32], [2 x i32]* %3, i64 0, i64 0
  %7 = call i32 @callee2(i32* noundef %6)
  ret i32 %7
}

define internal i32 @callee3(i32* noundef %0) {
; CHECK-LABEL: define {{[^@]+}}@callee3
; CHECK-SAME: (i32* noundef [[P_0:%.*]]) {
; CHECK-NEXT:    [[PL_0:%.*]] = getelementptr inbounds i32, i32* [[P_0]], i64 0
; CHECK-NEXT:    [[VAL_0:%.*]] = load i32, i32* [[PL_0]], align 4
; CHECK-NEXT:    [[PL_1:%.*]] = getelementptr inbounds i32, i32* [[P_0]], i64 1
; CHECK-NEXT:    [[VAL_1:%.*]] = load i32, i32* [[PL_1]], align 4
; CHECK-NEXT:    [[SUM_0:%.*]] = add nsw i32 [[VAL_0]], [[VAL_1]]
; CHECK-NEXT:    [[PL_2:%.*]] = getelementptr inbounds i32, i32* [[P_0]], i64 2
; CHECK-NEXT:    [[VAL_2:%.*]] = load i32, i32* [[PL_2]], align 4
; CHECK-NEXT:    [[RES:%.*]] = add nsw i32 [[SUM_0]], [[VAL_2]]
; CHECK-NEXT:    ret i32 [[RES]]
;
  %2 = getelementptr inbounds i32, i32* %0, i64 0
  %3 = load i32, i32* %2, align 4
  %4 = getelementptr inbounds i32, i32* %0, i64 1
  %5 = load i32, i32* %4, align 4
  %6 = add nsw i32 %3, %5
  %7 = getelementptr inbounds i32, i32* %0, i64 2
  %8 = load i32, i32* %7, align 4
  %9 = add nsw i32 %6, %8
  ret i32 %9
}

define i32 @caller3(i32 %0, i32 %1, i32 %2) {
; CHECK-LABEL: define {{[^@]+}}@caller3
; CHECK-SAME: (i32 [[P_0:%.*]], i32 [[P_1:%.*]], i32 [[P_2:%.*]]) {
; CHECK-NEXT:    [[TMP1:%.*]] = alloca [3 x i32], align 4
; CHECK-NEXT:    [[PL_0:%.*]] = getelementptr inbounds [3 x i32], [3 x i32]* [[TMP1]], i64 0, i64 0
; CHECK-NEXT:    store i32 [[P_0]], i32* [[PL_0]], align 4
; CHECK-NEXT:    [[PL_1:%.*]] = getelementptr inbounds i32, i32* [[PL_0]], i64 1
; CHECK-NEXT:    store i32 [[P_1]], i32* [[PL_1]], align 4
; CHECK-NEXT:    [[PL_2:%.*]] = getelementptr inbounds i32, i32* [[PL_1]], i64 1
; CHECK-NEXT:    store i32 [[P_2]], i32* [[PL_2]], align 4
; CHECK-NEXT:    [[PL_3:%.*]] = getelementptr inbounds [3 x i32], [3 x i32]* [[TMP1]], i64 0, i64 0
; CHECK-NEXT:    [[RES:%.*]] = call i32 @callee3(i32* noundef [[PL_3]])
; CHECK-NEXT:    ret i32 [[RES]]
;
  %4 = alloca [3 x i32], align 4
  %5 = getelementptr inbounds [3 x i32], [3 x i32]* %4, i64 0, i64 0
  store i32 %0, i32* %5, align 4
  %6 = getelementptr inbounds i32, i32* %5, i64 1
  store i32 %1, i32* %6, align 4
  %7 = getelementptr inbounds i32, i32* %6, i64 1
  store i32 %2, i32* %7, align 4
  %8 = getelementptr inbounds [3 x i32], [3 x i32]* %4, i64 0, i64 0
  %9 = call i32 @callee3(i32* noundef %8)
  ret i32 %9
}
