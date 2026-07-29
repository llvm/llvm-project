; Test that the min (-asan-mapping-min) and max (-asan-mapping-max) command-line options work as expected
;
; RUN: opt < %s -passes=asan -asan-mapping-min 0x1000 -S | FileCheck --check-prefix=CHECK-MIN %s
; RUN: opt < %s -passes=asan -asan-mapping-max 0x2000 -S | FileCheck --check-prefix=CHECK-MAX %s
; RUN: opt < %s -passes=asan -asan-mapping-min 0x1000 -asan-mapping-max 0x2000 -S | FileCheck --check-prefix=CHECK-BOTH %s
target triple = "x86_64-unknown-linux-gnu"

define i32 @read(ptr %a) sanitize_address {
entry:
  %tmp1 = load i32, ptr %a, align 4
  ret i32 %tmp1
}

define void @write(ptr %a, i32 %v) sanitize_address {
entry:
  store i32 %v, ptr %a, align 4
  ret void
}

define i32 @no_sanitize(ptr %a) {
entry:
  %tmp1 = load i32, ptr %a, align 4
  ret i32 %tmp1
}

; CHECK-MIN-LABEL: @read
; CHECK-MIN: [[ADDR:%[0-9]+]] = ptrtoaddr ptr %a to i64
; CHECK-MIN: [[CMP:%[0-9]+]] = icmp uge i64 [[ADDR]], 4096
; CHECK-MIN: br i1 [[CMP]], label %[[BEFORE_ASAN:[0-9]+]], label %[[AFTER_ASAN:[0-9]+]]
; CHECK-MIN: [[BEFORE_ASAN]]:
; CHECK-MIN: call void @__asan_report_load4
; CHECK-MIN: [[AFTER_ASAN]]:
; CHECK-MIN: %tmp1 = load i32, ptr %a, align 4

; CHECK-MIN-LABEL: @write
; CHECK-MIN: [[ADDR:%[0-9]+]] = ptrtoaddr ptr %a to i64
; CHECK-MIN: [[CMP:%[0-9]+]] = icmp uge i64 [[ADDR]], 4096
; CHECK-MIN: br i1 [[CMP]], label %[[BEFORE_ASAN:[0-9]+]], label %[[AFTER_ASAN:[0-9]+]]
; CHECK-MIN: [[BEFORE_ASAN]]:
; CHECK-MIN: call void @__asan_report_store4
; CHECK-MIN: [[AFTER_ASAN]]:
; CHECK-MIN: store i32 %v, ptr %a, align 4

; CHECK-MIN-LABEL: @no_sanitize
; CHECK-MIN-NOT: icmp uge i64
; CHECK-MIN-NOT: __asan_report
; CHECK-MIN: %tmp1 = load i32, ptr %a, align 4
; CHECK-MIN: ret i32 %tmp1


; CHECK-MAX-LABEL: @read
; CHECK-MAX: [[ADDR:%[0-9]+]] = ptrtoaddr ptr %a to i64
; CHECK-MAX: [[CMP:%[0-9]+]] = icmp ult i64 [[ADDR]], 8192
; CHECK-MAX: br i1 [[CMP]], label %[[BEFORE_ASAN:[0-9]+]], label %[[AFTER_ASAN:[0-9]+]]
; CHECK-MAX: [[BEFORE_ASAN]]:
; CHECK-MAX: call void @__asan_report_load4
; CHECK-MAX: [[AFTER_ASAN]]:
; CHECK-MAX: %tmp1 = load i32, ptr %a, align 4

; CHECK-MAX-LABEL: @write
; CHECK-MAX: [[ADDR:%[0-9]+]] = ptrtoaddr ptr %a to i64
; CHECK-MAX: [[CMP:%[0-9]+]] = icmp ult i64 [[ADDR]], 8192
; CHECK-MAX: br i1 [[CMP]], label %[[BEFORE_ASAN:[0-9]+]], label %[[AFTER_ASAN:[0-9]+]]
; CHECK-MAX: [[BEFORE_ASAN]]:
; CHECK-MAX: call void @__asan_report_store4
; CHECK-MAX: [[AFTER_ASAN]]:
; CHECK-MAX: store i32 %v, ptr %a, align 4

; CHECK-MAX-LABEL: @no_sanitize
; CHECK-MAX-NOT: icmp ult i64
; CHECK-MAX-NOT: __asan_report
; CHECK-MAX: %tmp1 = load i32, ptr %a, align 4
; CHECK-MAX: ret i32 %tmp1


; CHECK-BOTH-LABEL: @read
; CHECK-BOTH: [[ADDR:%[0-9]+]] = ptrtoaddr ptr %a to i64
; CHECK-BOTH: [[CMP_MIN:%[0-9]+]] = icmp uge i64 [[ADDR]], 4096
; CHECK-BOTH: br i1 [[CMP_MIN]], label %[[MIN_THEN:[0-9]+]], label %[[EXIT:[0-9]+]]
; CHECK-BOTH: [[MIN_THEN]]:
; CHECK-BOTH: [[ADDR2:%[0-9]+]] = ptrtoaddr ptr %a to i64
; CHECK-BOTH: [[CMP_MAX:%[0-9]+]] = icmp ult i64 [[ADDR2]], 8192
; CHECK-BOTH: br i1 [[CMP_MAX]], label %[[MAX_THEN:[0-9]+]], label %[[MIN_EXIT:[0-9]+]]
; CHECK-BOTH: [[MAX_THEN]]:
; CHECK-BOTH: call void @__asan_report_load4
; CHECK-BOTH: [[MIN_EXIT]]:
; CHECK-BOTH: br label %[[EXIT]]
; CHECK-BOTH: [[EXIT]]:
; CHECK-BOTH: %tmp1 = load i32, ptr %a, align 4

; CHECK-BOTH-LABEL: @write
; CHECK-BOTH: [[ADDR:%[0-9]+]] = ptrtoaddr ptr %a to i64
; CHECK-BOTH: [[CMP_MIN:%[0-9]+]] = icmp uge i64 [[ADDR]], 4096
; CHECK-BOTH: br i1 [[CMP_MIN]], label %[[MIN_THEN:[0-9]+]], label %[[EXIT:[0-9]+]]
; CHECK-BOTH: [[MIN_THEN]]:
; CHECK-BOTH: [[ADDR2:%[0-9]+]] = ptrtoaddr ptr %a to i64
; CHECK-BOTH: [[CMP_MAX:%[0-9]+]] = icmp ult i64 [[ADDR2]], 8192
; CHECK-BOTH: br i1 [[CMP_MAX]], label %[[MAX_THEN:[0-9]+]], label %[[MIN_EXIT:[0-9]+]]
; CHECK-BOTH: [[MAX_THEN]]:
; CHECK-BOTH: call void @__asan_report_store4
; CHECK-BOTH: [[MIN_EXIT]]:
; CHECK-BOTH: br label %[[EXIT]]
; CHECK-BOTH: [[EXIT]]:
; CHECK-BOTH: store i32 %v, ptr %a, align 4

; CHECK-BOTH-LABEL: @no_sanitize
; CHECK-BOTH-NOT: icmp uge i64
; CHECK-BOTH-NOT: icmp ult i64
; CHECK-BOTH-NOT: __asan_report
; CHECK-BOTH: %tmp1 = load i32, ptr %a, align 4
; CHECK-BOTH: ret i32 %tmp1
