; RUN: opt -passes=aggressive-instcombine -S < %s | FileCheck %s --check-prefix=AIC
; RUN: opt -passes='aggressive-instcombine,instcombine' -S < %s | FileCheck %s --check-prefix=COMBINED

declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg)

define void @range_0_1(ptr %dst, i8 %value, i64 %n) {
; AIC-LABEL: define void @range_0_1(
; AIC-SAME: ptr [[DST:%.*]], i8 [[VALUE:%.*]], i64 [[N:%.*]]) {
; AIC:       entry:
; AIC-NEXT:    [[LEN:%.*]] = and i64 [[N]], 1
; AIC-NEXT:    [[MEMSET_NOTZERO:%.*]] = icmp ne i64 [[LEN]], 0
; AIC-NEXT:    br i1 [[MEMSET_NOTZERO]], label %[[DO_MEMSET:.*]], label %[[END:.*]]
; AIC:       [[DO_MEMSET]]:
; AIC-NEXT:    call void @llvm.memset.p0.i64(ptr align 1 [[DST]], i8 [[VALUE]], i64 1, i1 false)
; AIC-NEXT:    br label %[[END]]
; AIC:       [[END]]:
; AIC-NEXT:    ret void
;
; COMBINED-LABEL: define void @range_0_1(
; COMBINED-SAME: ptr [[DST:%.*]], i8 [[VALUE:%.*]], i64 [[N:%.*]]) {
; COMBINED:       [[LEN:%.*]] = and i64 [[N]], 1
; COMBINED:       icmp {{eq|ne}} i64 [[LEN]], 0
; COMBINED:       br i1
; COMBINED:       store i8 [[VALUE]], ptr [[DST]], align 1
; COMBINED-NOT:   call void @llvm.memset
; COMBINED:       ret void
entry:
  %len = and i64 %n, 1
  call void @llvm.memset.p0.i64(ptr align 1 %dst, i8 %value, i64 %len, i1 false)
  ret void
}

define void @range_0_1_zext(ptr %dst, i8 %value, i32 %n) {
; AIC-LABEL: define void @range_0_1_zext(
; AIC-SAME: ptr [[DST:%.*]], i8 [[VALUE:%.*]], i32 [[N:%.*]]) {
; AIC:       entry:
; AIC-NEXT:    [[MASKED:%.*]] = and i32 [[N]], 1
; AIC-NEXT:    [[LEN:%.*]] = zext i32 [[MASKED]] to i64
; AIC-NEXT:    [[MEMSET_NOTZERO:%.*]] = icmp ne i64 [[LEN]], 0
; AIC-NEXT:    br i1 [[MEMSET_NOTZERO]], label %[[DO_MEMSET:.*]], label %[[END:.*]]
; AIC:       [[DO_MEMSET]]:
; AIC-NEXT:    call void @llvm.memset.p0.i64(ptr align 1 [[DST]], i8 [[VALUE]], i64 1, i1 false)
; AIC-NEXT:    br label %[[END]]
; AIC:       [[END]]:
; AIC-NEXT:    ret void
;
; COMBINED-LABEL: define void @range_0_1_zext(
; COMBINED-SAME: ptr [[DST:%.*]], i8 [[VALUE:%.*]], i32 [[N:%.*]]) {
; COMBINED:       [[MASKED:%.*]] = and i32 [[N]], 1
; COMBINED:       icmp {{eq|ne}} i32 [[MASKED]], 0
; COMBINED:       br i1
; COMBINED:       store i8 [[VALUE]], ptr [[DST]], align 1
; COMBINED-NOT:   call void @llvm.memset
; COMBINED:       ret void
entry:
  %masked = and i32 %n, 1
  %len = zext i32 %masked to i64
  call void @llvm.memset.p0.i64(ptr align 1 %dst, i8 %value, i64 %len, i1 false)
  ret void
}

define void @range_0_1_volatile(ptr %dst, i8 %value, i64 %n) {
; AIC-LABEL: define void @range_0_1_volatile(
; AIC-SAME: ptr [[DST:%.*]], i8 [[VALUE:%.*]], i64 [[N:%.*]]) {
; AIC:       entry:
; AIC-NEXT:    [[LEN:%.*]] = and i64 [[N]], 1
; AIC-NEXT:    [[MEMSET_NOTZERO:%.*]] = icmp ne i64 [[LEN]], 0
; AIC-NEXT:    br i1 [[MEMSET_NOTZERO]], label %[[DO_MEMSET:.*]], label %[[END:.*]]
; AIC:       [[DO_MEMSET]]:
; AIC-NEXT:    call void @llvm.memset.p0.i64(ptr align 1 [[DST]], i8 [[VALUE]], i64 1, i1 true)
; AIC-NEXT:    br label %[[END]]
; AIC:       [[END]]:
; AIC-NEXT:    ret void
;
; COMBINED-LABEL: define void @range_0_1_volatile(
; COMBINED-SAME: ptr [[DST:%.*]], i8 [[VALUE:%.*]], i64 [[N:%.*]]) {
; COMBINED:       [[LEN:%.*]] = and i64 [[N]], 1
; COMBINED:       icmp {{eq|ne}} i64 [[LEN]], 0
; COMBINED:       br i1
; COMBINED:       call void @llvm.memset.p0.i64(ptr align 1 [[DST]], i8 [[VALUE]], i64 1, i1 true)
; COMBINED:       ret void
entry:
  %len = and i64 %n, 1
  call void @llvm.memset.p0.i64(ptr align 1 %dst, i8 %value, i64 %len, i1 true)
  ret void
}

define void @range_0_2(ptr %dst, i8 %value, i64 %n) {
; AIC-LABEL: define void @range_0_2(
; AIC-SAME: ptr [[DST:%.*]], i8 [[VALUE:%.*]], i64 [[N:%.*]]) {
; AIC:       entry:
; AIC-NEXT:    [[LEN:%.*]] = urem i64 [[N]], 3
; AIC-NEXT:    call void @llvm.memset.p0.i64(ptr align 1 [[DST]], i8 [[VALUE]], i64 [[LEN]], i1 false)
; AIC-NEXT:    ret void
;
; COMBINED-LABEL: define void @range_0_2(
; COMBINED-SAME: ptr [[DST:%.*]], i8 [[VALUE:%.*]], i64 [[N:%.*]]) {
; COMBINED:       [[LEN:%.*]] = urem i64 [[N]], 3
; COMBINED-NEXT:  call void @llvm.memset.p0.i64(ptr align 1 [[DST]], i8 [[VALUE]], i64 [[LEN]], i1 false)
; COMBINED-NEXT:  ret void
entry:
  %len = urem i64 %n, 3
  call void @llvm.memset.p0.i64(ptr align 1 %dst, i8 %value, i64 %len, i1 false)
  ret void
}
