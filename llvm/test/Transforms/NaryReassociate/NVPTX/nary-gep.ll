; RUN: opt < %s -nary-reassociate -early-cse -earlycse-debug-hash -S | FileCheck %s
; RUN: opt < %s -passes='nary-reassociate' -S | opt -early-cse -S | FileCheck %s

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-unknown-unknown"

declare void @foo(ptr)

; foo(&a[i]);
; foo(&a[i + j]);
;   =>
; t = &a[i];
; foo(t);
; foo(t + j);
define void @reassociate_gep(ptr %a, i64 %i, i64 %j) {
; CHECK-LABEL: @reassociate_gep(
  %1 = add i64 %i, %j
  %2 = getelementptr float, ptr %a, i64 %i
; CHECK: [[t1:[^ ]+]] = getelementptr float, ptr %a, i64 %i
  call void @foo(ptr %2)
; CHECK: call void @foo(ptr [[t1]])
  %3 = getelementptr float, ptr %a, i64 %1
; CHECK: [[t2:[^ ]+]] = getelementptr float, ptr [[t1]], i64 %j
  call void @foo(ptr %3)
; CHECK: call void @foo(ptr [[t2]])
  ret void
}

; foo(&a[sext(j)]);
; foo(&a[sext(i +nsw j)]);
; foo(&a[sext((i +nsw j) +nsw i)]);
;   =>
; t1 = &a[sext(j)];
; foo(t1);
; t2 = t1 + sext(i);
; foo(t2);
; t3 = t2 + sext(i); // sext(i) should be GVN'ed.
; foo(t3);
define void @reassociate_gep_nsw(ptr %a, i32 %i, i32 %j) {
; CHECK-LABEL: @reassociate_gep_nsw(
  %idxprom.j = sext i32 %j to i64
  %1 = getelementptr float, ptr %a, i64 %idxprom.j
; CHECK: [[t1:[^ ]+]] = getelementptr float, ptr %a, i64 %idxprom.j
  call void @foo(ptr %1)
; CHECK: call void @foo(ptr [[t1]])

  %2 = add nsw i32 %i, %j
  %idxprom.2 = sext i32 %2 to i64
  %3 = getelementptr float, ptr %a, i64 %idxprom.2
; CHECK: [[sexti:[^ ]+]] = sext i32 %i to i64
; CHECK: [[t2:[^ ]+]] = getelementptr float, ptr [[t1]], i64 [[sexti]]
  call void @foo(ptr %3)
; CHECK: call void @foo(ptr [[t2]])

  %4 = add nsw i32 %2, %i
  %idxprom.4 = sext i32 %4 to i64
  %5 = getelementptr float, ptr %a, i64 %idxprom.4
; CHECK: [[t3:[^ ]+]] = getelementptr float, ptr [[t2]], i64 [[sexti]]
  call void @foo(ptr %5)
; CHECK: call void @foo(ptr [[t3]])

  ret void
}

; assume(j >= 0);
; foo(&a[zext(j)]);
; assume(i + j >= 0);
; foo(&a[zext(i + j)]);
;   =>
; t1 = &a[zext(j)];
; foo(t1);
; t2 = t1 + sext(i);
; foo(t2);
define void @reassociate_gep_assume(ptr %a, i32 %i, i32 %j) {
; CHECK-LABEL: @reassociate_gep_assume(
  ; assume(j >= 0)
  %cmp = icmp sgt i32 %j, -1
  call void @llvm.assume(i1 %cmp)
  %1 = add i32 %i, %j
  %cmp2 = icmp sgt i32 %1, -1
  call void @llvm.assume(i1 %cmp2)

  %idxprom.j = zext i32 %j to i64
  %2 = getelementptr float, ptr %a, i64 %idxprom.j
; CHECK: [[t1:[^ ]+]] = getelementptr float, ptr %a, i64 %idxprom.j
  call void @foo(ptr %2)
; CHECK: call void @foo(ptr [[t1]])

  %idxprom.1 = zext i32 %1 to i64
  %3 = getelementptr float, ptr %a, i64 %idxprom.1
; CHECK: [[sexti:[^ ]+]] = sext i32 %i to i64
; CHECK: [[t2:[^ ]+]] = getelementptr float, ptr [[t1]], i64 [[sexti]]
  call void @foo(ptr %3)
; CHECK: call void @foo(ptr [[t2]])

  ret void
}

; Do not split the second GEP because sext(i + j) != sext(i) + sext(j).
define void @reassociate_gep_no_nsw(ptr %a, i32 %i, i32 %j) {
; CHECK-LABEL: @reassociate_gep_no_nsw(
  %1 = add i32 %i, %j
  %2 = getelementptr float, ptr %a, i32 %j
; CHECK: getelementptr float, ptr %a, i32 %j
  call void @foo(ptr %2)
  %3 = getelementptr float, ptr %a, i32 %1
; CHECK: getelementptr float, ptr %a, i32 %1
  call void @foo(ptr %3)
  ret void
}

define void @reassociate_gep_128(ptr %a, i128 %i, i128 %j) {
; CHECK-LABEL: @reassociate_gep_128(
  %1 = add i128 %i, %j
  %2 = getelementptr float, ptr %a, i128 %i
; CHECK: [[t1:[^ ]+]] = getelementptr float, ptr %a, i128 %i
  call void @foo(ptr %2)
; CHECK: call void @foo(ptr [[t1]])
  %3 = getelementptr float, ptr %a, i128 %1
; CHECK: [[truncj:[^ ]+]] = trunc i128 %j to i64
; CHECK: [[t2:[^ ]+]] = getelementptr float, ptr [[t1]], i64 [[truncj]]
  call void @foo(ptr %3)
; CHECK: call void @foo(ptr [[t2]])
  ret void
}

%struct.complex = type { float, float }

declare void @bar(ptr)

define void @different_types(ptr %input, i64 %i) {
; CHECK-LABEL: @different_types(
; CHECK-NEXT: %t1 = getelementptr %struct.complex, ptr %input, i64 %i
; CHECK-NEXT: call void @bar(ptr %t1)
; CHECK-NEXT: %t2 = getelementptr float, ptr %t1, i64 10
; CHECK-NEXT: call void @foo(ptr %t2)
  %t1 = getelementptr %struct.complex, ptr %input, i64 %i
  call void @bar(ptr %t1)
  %j = add i64 %i, 5
  %t2 = getelementptr %struct.complex, ptr %input, i64 %j, i32 0
  call void @foo(ptr %t2)
  ret void
}

declare void @llvm.assume(i1)
