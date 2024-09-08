; RUN: opt -passes='loop-idiom' < %s -S | FileCheck %s

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

define i64 @valid_strlen_i8_test1(ptr %Str) {
; CHECK-LABEL: @valid_strlen_i8_test1(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TOBOOL:%.*]] = icmp eq ptr [[STR:%.*]], null
; CHECK-NEXT:    br i1 [[TOBOOL]], label [[CLEANUP:%.*]], label [[LOR_LHS_FALSE:%.*]]
; CHECK:       lor.lhs.false:
; CHECK-NEXT:    [[TMP0:%.*]] = load i8, ptr [[STR]], align 1
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i8 [[TMP0]], 0
; CHECK-NEXT:    br i1 [[CMP]], label [[CLEANUP]], label [[FOR_INC_PREHEADER:%.*]]
; CHECK:       for.inc.preheader:
; CHECK-NEXT:    [[SCEVGEP:%.*]] = getelementptr i8, ptr [[STR]], i64 0
; CHECK-NEXT:    [[STRLEN:%.*]] = call i64 @strlen(ptr [[SCEVGEP]])
; CHECK-NEXT:    br label [[FOR_INC:%.*]]
; CHECK:       for.inc:
; CHECK-NEXT:    [[SRC_09:%.*]] = phi ptr [ poison, [[FOR_INC]] ], [ [[STR]], [[FOR_INC_PREHEADER]] ]
; CHECK-NEXT:    [[TOBOOL2:%.*]] = icmp eq i8 poison, 0
; CHECK-NEXT:    br i1 true, label [[FOR_END:%.*]], label [[FOR_INC]]
; CHECK:       for.end:
; CHECK-NEXT:    br label [[CLEANUP]]
; CHECK:       cleanup:
; CHECK-NEXT:    [[RETVAL_0:%.*]] = phi i64 [ [[STRLEN]], [[FOR_END]] ], [ 0, [[ENTRY:%.*]] ], [ 0, [[LOR_LHS_FALSE]] ]
; CHECK-NEXT:    ret i64 [[RETVAL_0]]
;
entry:
  %tobool = icmp eq ptr %Str, null
  br i1 %tobool, label %cleanup, label %lor.lhs.false

lor.lhs.false:                                    ; preds = %entry
  %0 = load i8, ptr %Str, align 1
  %cmp = icmp eq i8 %0, 0
  br i1 %cmp, label %cleanup, label %for.inc

for.inc:                                          ; preds = %lor.lhs.false, %for.inc
  %Src.09 = phi ptr [ %incdec.ptr, %for.inc ], [ %Str, %lor.lhs.false ]
  %incdec.ptr = getelementptr inbounds i8, ptr %Src.09, i64 1
  %.pr = load i8, ptr %incdec.ptr, align 1
  %tobool2 = icmp eq i8 %.pr, 0
  br i1 %tobool2, label %for.end, label %for.inc

for.end:                                          ; preds = %for.inc
  %sub.ptr.lhs.cast = ptrtoint ptr %incdec.ptr to i64
  %sub.ptr.rhs.cast = ptrtoint ptr %Str to i64
  %sub.ptr.sub = sub i64 %sub.ptr.lhs.cast, %sub.ptr.rhs.cast
  br label %cleanup

cleanup:                                          ; preds = %lor.lhs.false, %entry, %for.end
  %retval.0 = phi i64 [ %sub.ptr.sub, %for.end ], [ 0, %entry ], [ 0, %lor.lhs.false ]
  ret i64 %retval.0
}

define i64 @valid_strlen_i8_test2(ptr %Str) {
; CHECK-LABEL: @valid_strlen_i8_test2(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TOBOOL:%.*]] = icmp eq ptr [[STR:%.*]], null
; CHECK-NEXT:    br i1 [[TOBOOL]], label [[CLEANUP:%.*]], label [[FOR_COND_PREHEADER:%.*]]
; CHECK:       for.cond.preheader:
; CHECK-NEXT:    [[STRLEN:%.*]] = call i64 @strlen(ptr [[STR]])
; CHECK-NEXT:    br label [[FOR_COND:%.*]]
; CHECK:       for.cond:
; CHECK-NEXT:    [[TOBOOL1:%.*]] = icmp eq i8 poison, 0
; CHECK-NEXT:    [[INCDEC_PTR:%.*]] = getelementptr inbounds i8, ptr poison, i64 1
; CHECK-NEXT:    br i1 true, label [[FOR_END:%.*]], label [[FOR_COND]]
; CHECK:       for.end:
; CHECK-NEXT:    br label [[CLEANUP]]
; CHECK:       cleanup:
; CHECK-NEXT:    [[RETVAL_0:%.*]] = phi i64 [ [[STRLEN]], [[FOR_END]] ], [ 0, [[ENTRY:%.*]] ]
; CHECK-NEXT:    ret i64 [[RETVAL_0]]
;
entry:
  %tobool = icmp eq ptr %Str, null
  br i1 %tobool, label %cleanup, label %for.cond

for.cond:                                         ; preds = %entry, %for.cond
  %Src.0 = phi ptr [ %incdec.ptr, %for.cond ], [ %Str, %entry ]
  %0 = load i8, ptr %Src.0, align 1
  %tobool1 = icmp eq i8 %0, 0
  %incdec.ptr = getelementptr inbounds i8, ptr %Src.0, i64 1
  br i1 %tobool1, label %for.end, label %for.cond

for.end:                                          ; preds = %for.cond
  %sub.ptr.lhs.cast = ptrtoint ptr %Src.0 to i64
  %sub.ptr.rhs.cast = ptrtoint ptr %Str to i64
  %sub.ptr.sub = sub i64 %sub.ptr.lhs.cast, %sub.ptr.rhs.cast
  br label %cleanup

  cleanup:                                          ; preds = %entry, %for.end
  %retval.0 = phi i64 [ %sub.ptr.sub, %for.end ], [ 0, %entry ]
  ret i64 %retval.0
}

define void @invalid_strlen_i8_test3(ptr %s, i32 zeroext %i) {
; CHECK-LABEL: @invalid_strlen_i8_test3(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[WHILE_COND:%.*]]
; CHECK:       while.cond:
; CHECK-NEXT:    [[S_ADDR_0:%.*]] = phi ptr [ [[S:%.*]], [[ENTRY:%.*]] ], [ [[INCDEC_PTR1:%.*]], [[WHILE_COND]] ]
; CHECK-NEXT:    [[TMP0:%.*]] = load i8, ptr [[S_ADDR_0]], align 1
; CHECK-NEXT:    [[TOBOOL_NOT:%.*]] = icmp eq i8 [[TMP0]], 0
; CHECK-NEXT:    [[INCDEC_PTR1]] = getelementptr inbounds i8, ptr [[S_ADDR_0]], i64 1
; CHECK-NEXT:    br i1 [[TOBOOL_NOT]], label [[WHILE_END:%.*]], label [[WHILE_COND]]
; CHECK:       while.end:
; CHECK-NEXT:    [[S_ADDR_0_LCSSA:%.*]] = phi ptr [ [[S_ADDR_0]], [[WHILE_COND]] ]
; CHECK-NEXT:    [[INCDEC_PTR1_LCSSA:%.*]] = phi ptr [ [[INCDEC_PTR1]], [[WHILE_COND]] ]
; CHECK-NEXT:    store i8 45, ptr [[S_ADDR_0_LCSSA]], align 1
; CHECK-NEXT:    [[CMP:%.*]] = icmp ult i32 [[I:%.*]], 10
; CHECK-NEXT:    br i1 [[CMP]], label [[IF_THEN:%.*]], label [[IF_END:%.*]]
; CHECK:       if.then:
; CHECK-NEXT:    store i8 65, ptr [[INCDEC_PTR1_LCSSA]], align 1
; CHECK-NEXT:    br label [[IF_END9:%.*]]
; CHECK:       if.end:
; CHECK-NEXT:    store i8 66, ptr [[INCDEC_PTR1_LCSSA]], align 1
; CHECK-NEXT:    br label [[IF_END9]]
; CHECK:       if.end9:
; CHECK-NEXT:    ret void
;
entry:
  br label %while.cond

while.cond:                                       ; preds = %while.cond, %entry
  %s.addr.0 = phi ptr [ %s, %entry ], [ %incdec.ptr1, %while.cond ]
  %0 = load i8, ptr %s.addr.0, align 1
  %tobool.not = icmp eq i8 %0, 0
  %incdec.ptr1 = getelementptr inbounds i8, ptr %s.addr.0, i64 1
  br i1 %tobool.not, label %while.end, label %while.cond

while.end:                                        ; preds = %while.cond
  %s.addr.0.lcssa = phi ptr [ %s.addr.0, %while.cond ]
  %incdec.ptr1.lcssa = phi ptr [ %incdec.ptr1, %while.cond ]
  store i8 45, ptr %s.addr.0.lcssa, align 1
  %cmp = icmp ult i32 %i, 10
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %while.end
  store i8 65, ptr %incdec.ptr1.lcssa, align 1
  br label %if.end9

if.end:                                           ; preds = %while.end
  store i8 66, ptr %incdec.ptr1.lcssa, align 1
  br label %if.end9

if.end9:                                          ; preds = %if.end, %if.then
  ret void
}

