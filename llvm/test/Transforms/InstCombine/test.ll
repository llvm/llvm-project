
define void @f2(ptr %a) {
; CHECK-LABEL: @f2(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @llvm.assume(i1 true) [ "align"(ptr [[A:%.*]], i64 32, i32 24) ]
; CHECK-NEXT:    [[TMP0:%.*]] = getelementptr inbounds nuw i8, ptr [[A]], i64 8
; CHECK-NEXT:    [[TMP1:%.*]] = ptrtoint ptr [[TMP0]] to i64
; CHECK-NEXT:    [[TMP2:%.*]] = and i64 [[TMP1]], 8
; CHECK-NEXT:    [[TMP3:%.*]] = icmp eq i64 [[TMP2]], 0
; CHECK-NEXT:    br i1 [[TMP3]], label [[IF_THEN:%.*]], label [[IF_ELSE:%.*]]
; CHECK:       if.then:
; CHECK-NEXT:    store i64 16, ptr [[TMP0]], align 4
; CHECK-NEXT:    br label [[IF_END:%.*]]
; CHECK:       if.else:
; CHECK-NEXT:    store i8 1, ptr [[TMP0]], align 1
; CHECK-NEXT:    br label [[IF_END]]
; CHECK:       if.end:
; CHECK-NEXT:    ret void
;
entry:
  call void @llvm.assume(i1 true) [ "align"(ptr %a, i64 32, i32 24) ]
  %0 = getelementptr inbounds i8, ptr %a, i64 8
  %1 = ptrtoint ptr %0 to i64
  %2 = and i64 %1, 15
  %3 = icmp eq i64 %2, 0
  br i1 %3, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  store i64 16, ptr %0, align 4
  br label %if.end

if.else:                                          ; preds = %entry
  store i8 1, ptr %0, align 1
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void
}
