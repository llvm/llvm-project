; RUN: opt -S -passes=instcombine,simplify-cfg < %s 2>&1 | FileCheck %s

declare void @llvm.assume(i1 noundef)

define void @f1(i8* %a) {
; CHECK-LABEL: @f1(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[PTR:%.*]] = getelementptr inbounds i8, i8* [[A:%.*]], i64 4
; CHECK-NEXT:    [[TMP0:%.*]] = ptrtoint i8* [[PTR]] to i64
; CHECK-NEXT:    [[TMP1:%.*]] = and i64 [[TMP0]], 3
; CHECK-NEXT:    [[TMP2:%.*]] = icmp eq i64 [[TMP1]], 0
; CHECK-NEXT:    br i1 [[TMP2]], label [[IF_THEN:%.*]], label [[IF_END:%.*]]
; CHECK:       if.then:
; CHECK-NEXT:    call void @llvm.assume(i1 true) [ "align"(i8* [[PTR]], i64 4) ]
; CHECK-NEXT:    [[TMP3:%.*]] = bitcast i8* [[PTR]] to i32*
; CHECK-NEXT:    store i32 4, i32* [[TMP3]], align 4
; CHECK-NEXT:    br label [[IF_END]]
; CHECK:       if.end:
; CHECK-NEXT:    ret void
;
entry:
  %ptr = getelementptr inbounds i8, i8* %a, i64 4
  %0 = ptrtoint i8* %ptr to i64
  %1 = and i64 %0, 3
  %2 = icmp eq i64 %1, 0
  br i1 %2, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @llvm.assume(i1 true) [ "align"(i8* %ptr, i64 4) ]
  %3 = ptrtoint i8* %ptr to i64
  %4 = and i64 %3, 3
  %5 = icmp eq i64 %4, 0
  br i1 %5, label %if.then1, label %if.else1

if.then1:                                         ; preds = %if.then
  %6 = bitcast i8* %ptr to i32*
  store i32 4, i32* %6, align 4
  br label %if.end

if.else1:                                         ; preds = %if.then
  store i8 1, i8* %ptr, align 1
  br label %if.end

if.end:                                           ; preds = %if.then1, %if.else1, %entry
  ret void
}

; TODO: We could fold away the branch "br i1 %3, ..." by either using a GEP or make getKnowledgeValidInContext aware the alignment bundle offset, and the improvement of value tracking of GEP. 

define void @f2(i8* %a) {
; CHECK-LABEL: @f2(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @llvm.assume(i1 true) [ "align"(i8* [[A:%.*]], i64 32, i32 24) ]
; CHECK-NEXT:    [[TMP0:%.*]] = getelementptr inbounds i8, i8* [[A]], i64 8
; CHECK-NEXT:    [[TMP1:%.*]] = ptrtoint i8* [[TMP0]] to i64
; CHECK-NEXT:    [[TMP2:%.*]] = and i64 [[TMP1]], 8
; CHECK-NEXT:    [[TMP3:%.*]] = icmp eq i64 [[TMP2]], 0
; CHECK-NEXT:    br i1 [[TMP3]], label [[IF_THEN1:%.*]], label [[IF_ELSE1:%.*]]
;
entry:
  call void @llvm.assume(i1 true) [ "align"(i8* %a, i64 32, i32 24) ]
  %0 = getelementptr inbounds i8, i8* %a, i64 8
  %1 = ptrtoint i8* %0 to i64
  %2 = and i64 %1, 15
  %3 = icmp eq i64 %2, 0
  br i1 %3, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %4 = bitcast i8* %0 to i64*
  store i64 16, i64* %4, align 4
  br label %if.end

if.else:                                          ; preds = %entry
  store i8 1, i8* %0, align 1
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void
}

