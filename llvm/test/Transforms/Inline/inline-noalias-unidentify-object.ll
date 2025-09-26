; RUN: opt < %s -passes=inline -enable-noalias-to-md-conversion -S | FileCheck %s --match-full-lines

define i32 @caller(ptr %p) {
; CHECK-LABEL: define i32 @caller(ptr %p) {
; CHECK-NEXT:    call void @llvm.experimental.noalias.scope.decl(metadata [[META0:![0-9]+]])
; CHECK-NEXT:    [[P_11_I:%.*]] = getelementptr i8, ptr %p, i64 11
; CHECK-NEXT:    [[V_I:%.*]] = load i32, ptr [[P_11_I]], align 4, !alias.scope !0
; CHECK-NEXT:    [[P_1_I:%.*]] = getelementptr i8, ptr %p, i64 1
; CHECK-NEXT:    [[P_2_I:%.*]] = getelementptr i8, ptr [[P_1_I]], i64 1
; CHECK-NEXT:    [[P_3_I:%.*]] = getelementptr i8, ptr [[P_2_I]], i64 1
; CHECK-NEXT:    [[P_4_I:%.*]] = getelementptr i8, ptr [[P_3_I]], i64 1
; CHECK-NEXT:    [[P_5_I:%.*]] = getelementptr i8, ptr [[P_4_I]], i64 1
; CHECK-NEXT:    [[P_6_I:%.*]] = getelementptr i8, ptr [[P_5_I]], i64 1
; CHECK-NEXT:    [[P_7_I1:%.*]] = getelementptr i8, ptr [[P_6_I]], i64 1
; CHECK-NEXT:    [[P_8_I:%.*]] = getelementptr i8, ptr [[P_7_I1]], i64 1
; CHECK-NEXT:    [[P_9_I:%.*]] = getelementptr i8, ptr [[P_8_I]], i64 1
; CHECK-NEXT:    [[P_7_I:%.*]] = getelementptr i8, ptr [[P_9_I]], i64 1
; CHECK-NEXT:    [[P_8_ALIAS_I:%.*]] = getelementptr i8, ptr [[P_7_I]], i64 1
; CHECK-NEXT:    store i32 42, ptr [[P_8_ALIAS_I]], align 4
; CHECK-NEXT:    ret i32 [[V_I]]
;
  %v = call i32 @callee(ptr %p)
  ret i32 %v
}

define internal i32 @callee(ptr noalias %p) {
  %p.11 = getelementptr i8, ptr %p, i64 11
  %v = load i32, ptr %p.11
  %p.1 = getelementptr i8, ptr %p, i64 1
  %p.2 = getelementptr i8, ptr %p.1, i64 1
  %p.3 = getelementptr i8, ptr %p.2, i64 1
  %p.4 = getelementptr i8, ptr %p.3, i64 1
  %p.5 = getelementptr i8, ptr %p.4, i64 1
  %p.6 = getelementptr i8, ptr %p.5, i64 1
  %p.7 = getelementptr i8, ptr %p.6, i64 1
  %p.8 = getelementptr i8, ptr %p.7, i64 1
  %p.9 = getelementptr i8, ptr %p.8, i64 1
  %p.10 = getelementptr i8, ptr %p.9, i64 1
  %p.11.alias = getelementptr i8, ptr %p.10, i64 1
  store i32 42, ptr %p.11.alias
  ret i32 %v
}
