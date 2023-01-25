; RUN: llvm-reduce --delta-passes=functions --test FileCheck --test-arg --check-prefixes=INTERESTING --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefixes=RESULT --input-file=%t %s

; FIXME: This testcase exhibits nonsensical behavior. The first
; function has blockaddress references. When the second function is
; deleted, it causes the blockreferences from the first to be replaced
; with inttoptr.

; INTERESTING: @blockaddr.table.other

; RESULT: @blockaddr.table.other = private unnamed_addr constant [2 x ptr] [ptr inttoptr (i32 1 to ptr), ptr inttoptr (i32 1 to ptr)]

@blockaddr.table.other = private unnamed_addr constant [2 x ptr] [ptr blockaddress(@bar, %L1), ptr blockaddress(@bar, %L2)]


; RESULT: define i32 @bar(
define i32 @bar(i64 %arg0) {
entry:
  %gep = getelementptr inbounds [2 x ptr], ptr @blockaddr.table.other, i64 0, i64 %arg0
  %load = load ptr, ptr %gep, align 8
  indirectbr ptr %load, [label %L2, label %L1]

L1:
  %phi = phi i32 [ 1, %L2 ], [ 2, %entry ]
  ret i32 %phi

L2:
  br label %L1
}

; RESULT-NOT: @unused
define void @unused() {
entry:
  br label %exit

exit:
  ret void
}
