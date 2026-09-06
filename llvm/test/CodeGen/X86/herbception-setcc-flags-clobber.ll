; RUN: llc -O2 -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s

; Herbception (throws): the carry-flag discriminant of a throws call is
; materialized by a HERB_SETCCr (pseudo setb) glued to the call. A peephole
; folds
;   %disc = herb_setcc; testb $1, %disc; jcc
; into a direct jb/jae on live CF. That fold is only valid when nothing
; between the HERB_SETCCr and the branch modifies EFLAGS. Here the loop
; countdown decrement gets scheduled between the call and the branch, so
; the fold must not happen: a folded jae would test the countdown borrow
; instead of the post-call error flag and corrupt the error check (this
; made a scatter-write loop exit after one iteration at -Oz). The
; HERB_SETCCr/TEST8ri pair must be kept and the discriminant tested from
; the register.

target triple = "x86_64-unknown-linux-gnu"

declare dso_local { { ptr, i64 }, i1 } @callee(ptr, ptr, ptr) #1

define dso_local { { ptr, i64 }, i1 } @caller(ptr %0, ptr noundef %1, i64 noundef %2) local_unnamed_addr #0 {
  %4 = tail call i64 @write(i32 noundef 2, ptr noundef nonnull null, i64 noundef 7) #19
  %5 = getelementptr inbounds nuw [16 x i8], ptr %1, i64 %2
  br label %6

6:                                                ; preds = %11, %3
  %7 = phi ptr [ %1, %3 ], [ %22, %11 ]
  %8 = phi ptr [ undef, %3 ], [ %21, %11 ]
  %9 = phi i64 [ undef, %3 ], [ %20, %11 ]
  %10 = icmp ne ptr %7, %5
  br i1 %10, label %11, label %23

11:                                               ; preds = %6
  %12 = load ptr, ptr %7, align 8
  %13 = getelementptr inbounds nuw i8, ptr %7, i64 8
  %14 = load i64, ptr %13, align 8
  %15 = tail call i64 @write(i32 noundef 2, ptr noundef nonnull null, i64 noundef 9) #19
  %16 = getelementptr inbounds nuw i8, ptr %12, i64 %14
  %17 = tail call { { ptr, i64 }, i1 } @callee(ptr %0, ptr noundef %12, ptr noundef %16) #1
  %18 = extractvalue { { ptr, i64 }, i1 } %17, 0
  %19 = extractvalue { { ptr, i64 }, i1 } %17, 1
  %20 = extractvalue { ptr, i64 } %18, 1
  %21 = extractvalue { ptr, i64 } %18, 0
  %22 = getelementptr inbounds nuw i8, ptr %7, i64 16
  br i1 %19, label %25, label %6

23:                                               ; preds = %6
  %24 = tail call i64 @write(i32 noundef 2, ptr noundef nonnull null, i64 noundef 8) #19
  br label %25

25:                                               ; preds = %11, %23
  %26 = phi i64 [ %9, %23 ], [ %20, %11 ]
  %27 = phi ptr [ %8, %23 ], [ %21, %11 ]
  %28 = insertvalue { { ptr, i64 }, i1 } poison, ptr %27, 0, 0
  %29 = insertvalue { { ptr, i64 }, i1 } %28, i64 %26, 0, 1
  %30 = insertvalue { { ptr, i64 }, i1 } %29, i1 %10, 1
  ret { { ptr, i64 }, i1 } %30
}


attributes #0 = { minsize optsize }
attributes #1 = { minsize nounwind optsize throws }
declare i64 @write(i32, ptr, i64)

; CHECK-LABEL: {{^}}caller:
; CHECK: callq callee
; CHECK: setb
; CHECK: testb $1
; CHECK: je
; CHECK-NOT: jae
