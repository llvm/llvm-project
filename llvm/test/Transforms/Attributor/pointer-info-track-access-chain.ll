; RUN: opt -aa-pipeline=basic-aa -passes=attributor -attributor-manifest-internal -debug-only=attributor  -attributor-annotate-decl-cs  -S < %s 2>&1 | FileCheck %s
; RUN: opt -aa-pipeline=basic-aa -passes=attributor-cgscc -attributor-manifest-internal -debug-only=attributor  -attributor-annotate-decl-cs -S < %s 2>&1  | FileCheck %s
; REQUIRES: asserts


@globalBytes = internal global [1024 x i8] zeroinitializer, align 16

; CHECK: Accesses by bin after update:
; CHECK: [8-12] : 1
; CHECK:      - 5 -   %1 = load i32, ptr %field22, align 4
; CHECK:        - c: <unknown>
; CHECK: Print all access paths found:
; CHECK: Backtrack a unique access path:
; CHECK:   %1 = load i32, ptr %field22, align 4
; CHECK:   %field22 = getelementptr i32, ptr %field2, i32 0
; CHECK:   %field2 = getelementptr inbounds [10 x i32], ptr %f, i32 0, i32 2
; CHECK:   %f = alloca [10 x i32], align 4
; CHECK: [4-5] : 1
; CHECK:      - 9 -   store i8 10, ptr %field11, align 4
; CHECK:        - c: i8 10
; CHECK: Print all access paths found:
; CHECK: Backtrack a unique access path:
; CHECK:   store i8 10, ptr %field11, align 4
; CHECK:   %field11 = getelementptr i32, ptr %field1, i32 0
; CHECK:   %field1 = getelementptr inbounds [10 x i32], ptr %f, i32 0, i32 1
; CHECK:   %f = alloca [10 x i32], align 4
; CHECK: [32-36] : 1
; CHECK:      - 9 -   store i32 %3, ptr %field8, align 4
; CHECK:        - c:   %3 = load i32, ptr %val, align 4
; CHECK: Print all access paths found:
; CHECK: Backtrack a unique access path:
; CHECK:   store i32 %3, ptr %field8, align 4
; CHECK:   %field8 = getelementptr inbounds [10 x i32], ptr %f, i32 0, i32 8
; CHECK:   %f = alloca [10 x i32], align 4
; CHECK: [4-8] : 1
; CHECK:      - 5 -   %0 = load i32, ptr %field11, align 4
; CHECK:        - c: <unknown>
; CHECK: Print all access paths found:
; CHECK: Backtrack a unique access path:
; CHECK:   %0 = load i32, ptr %field11, align 4
; CHECK:   %field11 = getelementptr i32, ptr %field1, i32 0
; CHECK:   %field1 = getelementptr inbounds [10 x i32], ptr %f, i32 0, i32 1
; CHECK:   %f = alloca [10 x i32], align 4
; CHECK: [8-9] : 1
; CHECK:      - 9 -   store i8 12, ptr %field22, align 4
; CHECK:        - c: i8 12
; CHECK: Print all access paths found:
; CHECK: Backtrack a unique access path:
; CHECK:   store i8 12, ptr %field22, align 4
; CHECK:   %field22 = getelementptr i32, ptr %field2, i32 0
; CHECK:   %field2 = getelementptr inbounds [10 x i32], ptr %f, i32 0, i32 2
; CHECK:   %f = alloca [10 x i32], align 4
define dso_local i32 @track_chain(ptr nocapture %val) #0 {
entry:
  %f = alloca [10 x i32]
  %field1 = getelementptr inbounds [10 x i32], ptr %f, i32 0, i32 1
  %field2 = getelementptr inbounds [10 x i32], ptr %f, i32 0, i32 2
  %field3 = getelementptr inbounds [10 x i32], ptr %f, i32 0, i32 3
  %field8 = getelementptr inbounds [10 x i32], ptr %f, i32 0, i32 8

  %field11 = getelementptr i32, ptr %field1, i32 0
  %field22 = getelementptr i32, ptr %field2, i32 0
  store i8 10, ptr %field11, align 4
  store i8 12, ptr %field22, align 4

  %1 = load i32, ptr %field11, align 4
  %2 = load i32, ptr %field22, align 4
  %3 = add i32 %1, %2

  %4 = load i32, ptr %val, align 4
  store i32 %4, ptr %field8, align 4

  %5 = add i32 %4, %3

  ret i32 %5
}


; CHECK: Accesses by bin after update:
; CHECK: [12-16] : 1
; CHECK:      - 5 -   %0 = load i32, ptr %field11, align 4
; CHECK:        - c: <unknown>
; CHECK: Print all access paths found:
; CHECK: Backtrack a unique access path:
; CHECK:   %0 = load i32, ptr %field11, align 4
; CHECK:   %field11 = getelementptr i32, ptr %field1, i32 2
; CHECK:   %field1 = getelementptr inbounds [10 x i32], ptr %f, i32 0, i32 1
; CHECK:   %f = alloca [10 x i32], align 4
; CHECK: [16-17] : 1
; CHECK:      - 9 -   store i8 12, ptr %field22, align 4
; CHECK:        - c: i8 12
; CHECK: Print all access paths found:
; CHECK: Backtrack a unique access path:
; CHECK:   store i8 12, ptr %field22, align 4
; CHECK:   %field22 = getelementptr i32, ptr %field2, i32 2
; CHECK:   %field2 = getelementptr inbounds [10 x i32], ptr %f, i32 0, i32 2
; CHECK:   %f = alloca [10 x i32], align 4
; CHECK: [32-36] : 1
; CHECK:      - 9 -   store i32 %3, ptr %field8, align 4
; CHECK:        - c:   %3 = load i32, ptr %val, align 4
; CHECK: Print all access paths found:
; CHECK: Backtrack a unique access path:
; CHECK:   store i32 %3, ptr %field8, align 4
; CHECK:   %field8 = getelementptr inbounds [10 x i32], ptr %f, i32 0, i32 8
; CHECK:   %f = alloca [10 x i32], align 4
; CHECK: [16-20] : 1
; CHECK:      - 5 -   %1 = load i32, ptr %field22, align 4
; CHECK:        - c: <unknown>
; CHECK: Print all access paths found:
; CHECK: Backtrack a unique access path:
; CHECK:   %1 = load i32, ptr %field22, align 4
; CHECK:   %field22 = getelementptr i32, ptr %field2, i32 2
; CHECK:   %field2 = getelementptr inbounds [10 x i32], ptr %f, i32 0, i32 2
; CHECK:   %f = alloca [10 x i32], align 4
; CHECK: [12-13] : 1
; CHECK:      - 9 -   store i8 10, ptr %field11, align 4
; CHECK:        - c: i8 10
; CHECK: Print all access paths found:
; CHECK: Backtrack a unique access path:
; CHECK:   store i8 10, ptr %field11, align 4
; CHECK:   %field11 = getelementptr i32, ptr %field1, i32 2
; CHECK:   %field1 = getelementptr inbounds [10 x i32], ptr %f, i32 0, i32 1
; CHECK:   %f = alloca [10 x i32], align 4
define dso_local i32 @track_chain_2(ptr nocapture %val) #0 {
entry:
  %f = alloca [10 x i32]
  %field1 = getelementptr inbounds [10 x i32], ptr %f, i32 0, i32 1
  %field2 = getelementptr inbounds [10 x i32], ptr %f, i32 0, i32 2
  %field3 = getelementptr inbounds [10 x i32], ptr %f, i32 0, i32 3
  %field8 = getelementptr inbounds [10 x i32], ptr %f, i32 0, i32 8

  %field11 = getelementptr i32, ptr %field1, i32 2
  %field22 = getelementptr i32, ptr %field2, i32 2
  store i8 10, ptr %field11, align 4
  store i8 12, ptr %field22, align 4

  %1 = load i32, ptr %field11, align 4
  %2 = load i32, ptr %field22, align 4
  %3 = add i32 %1, %2

  %4 = load i32, ptr %val, align 4
  store i32 %4, ptr %field8, align 4

  %5 = add i32 %4, %3

  ret i32 %5
}


; CHECK: Accesses by bin after update:
; CHECK: [12-16] : 3
; CHECK:      - 5 -   %0 = load i32, ptr %field11, align 4
; CHECK:       - c: <unknown>
; CHECK: Print all access paths found:
; CHECK: Backtrack a unique access path:
; CHECK:   %0 = load i32, ptr %field11, align 4
; CHECK:   %field11 = getelementptr i32, ptr %field1, i32 2
; CHECK:   %field1 = getelementptr inbounds [10 x i32], ptr %f, i32 0, i32 1
; CHECK:   %f = alloca [10 x i32], align 4
; CHECK:      - 5 -   %b = load i32, ptr %field3, align 4
; CHECK:        - c: <unknown>
; CHECK: Print all access paths found:
; CHECK: Backtrack a unique access path:
; CHECK:   %b = load i32, ptr %field3, align 4
; CHECK:   %field3 = getelementptr inbounds [10 x i32], ptr %f, i32 0, i32 3
; CHECK:   %f = alloca [10 x i32], align 4
; CHECK:      - 10 -   store i32 1000, ptr %6, align 4
; CHECK:        - c: i32 1000
; CHECK: Print all access paths found:
; CHECK: Backtrack a unique access path:
; CHECK:   store i32 1000, ptr %6, align 4
; CHECK:   %6 = select i1 %cond, ptr %field3, ptr %field8
; CHECK:   %field3 = getelementptr inbounds [10 x i32], ptr %f, i32 0, i32 3
; CHECK:   %f = alloca [10 x i32], align 4
; CHECK: Backtrack a unique access path:
; CHECK:   store i32 1000, ptr %6, align 4
; CHECK:   %6 = select i1 %cond, ptr %field3, ptr %field8
; CHECK:   %field8 = getelementptr inbounds [10 x i32], ptr %f, i32 0, i32 8
; CHECK:   %f = alloca [10 x i32], align 4
; CHECK: [16-17] : 1
; CHECK:      - 9 -   store i8 12, ptr %field22, align 4
; CHECK:       - c: i8 12
; CHECK: Print all access paths found:
; CHECK: Backtrack a unique access path:
; CHECK:   store i8 12, ptr %field22, align 4
; CHECK:   %field22 = getelementptr i32, ptr %field2, i32 2
; CHECK:   %field2 = getelementptr inbounds [10 x i32], ptr %f, i32 0, i32 2
; CHECK:   %f = alloca [10 x i32], align 4
; CHECK: [32-36] : 4
; CHECK:      - 9 -   store i32 %3, ptr %field8, align 4
; CHECK:        - c:   %3 = load i32, ptr %val, align 4
; CHECK: Print all access paths found:
; CHECK: Backtrack a unique access path:
; CHECK:   store i32 %3, ptr %field8, align 4
; CHECK:   %field8 = getelementptr inbounds [10 x i32], ptr %f, i32 0, i32 8
; CHECK:   %f = alloca [10 x i32], align 4
; CHECK:      - 5 -   %a1 = load i32, ptr %field8, align 4
; CHECK:        - c: <unknown>
; CHECK: Print all access paths found:
; CHECK: Backtrack a unique access path:
; CHECK:   %a1 = load i32, ptr %field8, align 4
; CHECK:   %field8 = getelementptr inbounds [10 x i32], ptr %f, i32 0, i32 8
; CHECK:   %f = alloca [10 x i32], align 4
; CHECK:      - 10 -   store i32 1000, ptr %6, align 4
; CHECK:        - c: i32 1000
; CHECK: Print all access paths found:
; CHECK: Backtrack a unique access path:
; CHECK:   store i32 1000, ptr %6, align 4
; CHECK:   %6 = select i1 %cond, ptr %field3, ptr %field8
; CHECK:   %field3 = getelementptr inbounds [10 x i32], ptr %f, i32 0, i32 3
; CHECK:   %f = alloca [10 x i32], align 4
; CHECK: Backtrack a unique access path:
; CHECK:   store i32 1000, ptr %6, align 4
; CHECK:   %6 = select i1 %cond, ptr %field3, ptr %field8
; CHECK:  %field8 = getelementptr inbounds [10 x i32], ptr %f, i32 0, i32 8
; CHECK:   %f = alloca [10 x i32], align 4
; CHECK:      - 5 -   %8 = load i32, ptr %field8, align 4
; CHECK:        - c: <unknown>
; CHECK: Print all access paths found:
; CHECK: Backtrack a unique access path:
; CHECK:  %8 = load i32, ptr %field8, align 4
; CHECK:  %field8 = getelementptr inbounds [10 x i32], ptr %f, i32 0, i32 8
; CHECK:  %f = alloca [10 x i32], align 4
; CHECK: [16-20] : 1
; CHECK:      - 5 -   %1 = load i32, ptr %field22, align 4
; CHECK:        - c: <unknown>
; CHECK: Print all access paths found:
; CHECK: Backtrack a unique access path:
; CHECK:   %1 = load i32, ptr %field22, align 4
; CHECK:   %field22 = getelementptr i32, ptr %field2, i32 2
; CHECK:   %field2 = getelementptr inbounds [10 x i32], ptr %f, i32 0, i32 2
; CHECK:   %f = alloca [10 x i32], align 4
; CHECK: [12-13] : 1
; CHECK:      - 9 -   store i8 10, ptr %field11, align 4
; CHECK:        - c: i8 10
; CHECK: Print all access paths found:
; CHECK: Backtrack a unique access path:
; CHECK:   store i8 10, ptr %field11, align 4
; CHECK:   %field11 = getelementptr i32, ptr %field1, i32 2
; CHECK:   %field1 = getelementptr inbounds [10 x i32], ptr %f, i32 0, i32 1
; CHECK:   %f = alloca [10 x i32], align 4
define dso_local i32 @track_chain_3(ptr nocapture %val, i1 %cond) #0 {
entry:
  %f = alloca [10 x i32]
  %field1 = getelementptr inbounds [10 x i32], ptr %f, i32 0, i32 1
  %field2 = getelementptr inbounds [10 x i32], ptr %f, i32 0, i32 2
  %field3 = getelementptr inbounds [10 x i32], ptr %f, i32 0, i32 3
  %field8 = getelementptr inbounds [10 x i32], ptr %f, i32 0, i32 8

  %field11 = getelementptr i32, ptr %field1, i32 2
  %field22 = getelementptr i32, ptr %field2, i32 2
  store i8 10, ptr %field11, align 4
  store i8 12, ptr %field22, align 4
  %1 = load i32, ptr %field11, align 4
  %2 = load i32, ptr %field22, align 4
  %3 = add i32 %1, %2
  %4 = load i32, ptr %val, align 4
  store i32 %4, ptr %field8, align 4
  %5 = add i32 %4, %3
  %6 = load i32, ptr %val
  %a1 = load i32, ptr %field8
  %a = add i32 %a1, %6
  %b = load i32, ptr %field3
  ;%b  = sub i32 %b1, %6
  %7 = select i1 %cond, ptr %field3, ptr %field8
  store i32 1000, ptr %7
  %8 = add i32 %5, %b
  %9 = load i32, ptr %field8
  %10 = add i32 %9, %8
  ret i32 %10
}

; CHECK: Accesses by bin after update:
; CHECK: [8-12] : 2
; CHECK:      - 9 -   store i32 %0, ptr %field2, align 4
; CHECK:        - c:   %0 = load i32, ptr %val, align 4
; CHECK: Print all access paths found:
; CHECK: Backtrack a unique access path:
; CHECK:   store i32 %0, ptr %field2, align 4
; CHECK:   %field2 = getelementptr i32, ptr @globalBytes, i32 2
; CHECK: @globalBytes = internal global [1024 x i8] zeroinitializer, align 16
; CHECK:      - 6 -   %ret = load i32, ptr %x, align 4
; CHECK:        - c: <unknown>
; CHECK: Print all access paths found:
; CHECK: Backtrack a unique access path:
; CHECK:   %ret = load i32, ptr %x, align 4
; CHECK:   %x = phi ptr [ %field2, %then ], [ %field8, %else ]
; CHECK:   %field2 = getelementptr i32, ptr @globalBytes, i32 2
; CHECK: @globalBytes = internal global [1024 x i8] zeroinitializer, align 16
; CHECK: Backtrack a unique access path:
; CHECK:   %ret = load i32, ptr %x, align 4
; CHECK:   %x = phi ptr [ %field2, %then ], [ %field8, %else ]
; CHECK:   %field8 = getelementptr i32, ptr @globalBytes, i32 8
; CHECK: @globalBytes = internal global [1024 x i8] zeroinitializer, align 16
; CHECK: [32-36] : 5
; CHECK:      - 6 -   %ret = load i32, ptr %x, align 4
; CHECK:        - c: <unknown>
; CHECK: Print all access paths found:
; CHECK: Backtrack a unique access path:
; CHECK:   %ret = load i32, ptr %x, align 4
; CHECK:   %x = phi ptr [ %field2, %then ], [ %field8, %else ]
; CHECK:   %field2 = getelementptr i32, ptr @globalBytes, i32 2
; CHECK: @globalBytes = internal global [1024 x i8] zeroinitializer, align 16
; CHECK: Backtrack a unique access path:
; CHECK:   %ret = load i32, ptr %x, align 4
; CHECK:   %x = phi ptr [ %field2, %then ], [ %field8, %else ]
; CHECK:   %field8 = getelementptr i32, ptr @globalBytes, i32 8
; CHECK: @globalBytes = internal global [1024 x i8] zeroinitializer, align 16
; CHECK:      - 9 -   store i32 %1, ptr %field8, align 4
; CHECK:        - c:   %1 = load i32, ptr %val2, align 4
; CHECK: Print all access paths found:
; CHECK: Backtrack a unique access path:
; CHECK:   store i32 %1, ptr %field8, align 4
; CHECK:   %field8 = getelementptr i32, ptr @globalBytes, i32 8
; CHECK: @globalBytes = internal global [1024 x i8] zeroinitializer, align 16
; CHECK:      - 9 -   store i32 %0, ptr %field2, align 4
; CHECK:        - c:   %0 = load i32, ptr %val, align 4
; CHECK: Print all access paths found:
; CHECK: Backtrack a unique access path:
; CHECK:   store i32 %0, ptr %field2, align 4
; CHECK:   %field2 = getelementptr i32, ptr @globalBytes, i32 8
; CHECK: @globalBytes = internal global [1024 x i8] zeroinitializer, align 16
; CHECK:      - 6 -   %ret = load i32, ptr %x, align 4
; CHECK:        - c: <unknown>
; CHECK: Print all access paths found:
; CHECK: Backtrack a unique access path:
; CHECK:   %ret = load i32, ptr %x, align 4
; CHECK:   %x = phi ptr [ %field2, %then ], [ %field8, %else ]
; CHECK:   %field2 = getelementptr i32, ptr @globalBytes, i32 8
; CHECK: @globalBytes = internal global [1024 x i8] zeroinitializer, align 16
; CHECK: Backtrack a unique access path:
; CHECK:   %ret = load i32, ptr %x, align 4
; CHECK:   %x = phi ptr [ %field2, %then ], [ %field8, %else ]
; CHECK:   %field8 = getelementptr i32, ptr @globalBytes, i32 8
; CHECK: @globalBytes = internal global [1024 x i8] zeroinitializer, align 16
; CHECK:      - 9 -   store i32 %1, ptr %field8, align 4
; CHECK:        - c:   %1 = load i32, ptr %val2, align 4
; CHECK: Print all access paths found:
; CHECK: Backtrack a unique access path:
; CHECK:   store i32 %1, ptr %field8, align 4
; CHECK:   %field8 = getelementptr i32, ptr @globalBytes, i32 8
; CHECK: @globalBytes = internal global [1024 x i8] zeroinitializer, align 16

define dso_local i32 @phi_different_offsets(ptr nocapture %val, ptr nocapture %val2, i1 %cmp) {
entry:
  br i1 %cmp, label %then, label %else

then:
  %field2 = getelementptr i32, ptr @globalBytes, i32 2
  %0 = load i32, ptr %val
  store i32 %0, ptr %field2
  br label %end

else:
  %field8 = getelementptr i32, ptr @globalBytes, i32 8
  %2 = load i32, ptr %val2
  store i32 %2, ptr %field8
  br label %end

end:
  %x = phi ptr [ %field2, %then ], [ %field8, %else ]
  %ret = load i32, ptr %x
  ret i32 %ret

}

define dso_local i32 @phi_same_offsets(ptr nocapture %val, ptr nocapture %val2, i1 %cmp) {
entry:
  br i1 %cmp, label %then, label %else

then:
  %field2 = getelementptr i32, ptr @globalBytes, i32 8
  %0 = load i32, ptr %val
  store i32 %0, ptr %field2
  br label %end

else:
  %field8 = getelementptr i32, ptr @globalBytes, i32 8
  %2 = load i32, ptr %val2
  store i32 %2, ptr %field8
  br label %end

end:
  %x = phi ptr [ %field2, %then ], [ %field8, %else ]
  %ret = load i32, ptr %x
  ret i32 %ret
}