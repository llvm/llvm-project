; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=functions --test FileCheck --test-arg --check-prefixes=INTERESTING --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefixes=RESULT --input-file=%t %s

; Make sure we don't crash on blockaddress
; TODO: Should be able to replace the blockaddresses with null too

; INTERESTING: @blockaddr.table
; INTERESTING: @blockaddr.table.addrspacecast

; RESULT: @blockaddr.table = private unnamed_addr constant [2 x ptr] [ptr blockaddress(@foo, %L1), ptr blockaddress(@foo, %L2)]
; RESULT: @blockaddr.table.addrspacecast = private unnamed_addr constant [2 x ptr addrspace(1)] [ptr addrspace(1) addrspacecast (ptr blockaddress(@foo_addrspacecast, %L1) to ptr addrspace(1)), ptr addrspace(1) addrspacecast (ptr blockaddress(@foo_addrspacecast, %L2) to ptr addrspace(1))]

@blockaddr.table = private unnamed_addr constant [2 x ptr] [ptr blockaddress(@foo, %L1), ptr blockaddress(@foo, %L2)]

@blockaddr.table.addrspacecast = private unnamed_addr constant [2 x ptr addrspace(1)] [
  ptr addrspace(1) addrspacecast (ptr blockaddress(@foo_addrspacecast, %L1) to ptr addrspace(1)),
  ptr addrspace(1) addrspacecast (ptr blockaddress(@foo_addrspacecast, %L2) to ptr addrspace(1))
]

; RESULT: define i32 @foo(
define i32 @foo(i64 %arg0) {
entry:
  %gep = getelementptr inbounds [2 x ptr], ptr @blockaddr.table, i64 0, i64 %arg0
  %load = load ptr, ptr %gep, align 8
  indirectbr ptr %load, [label %L2, label %L1]

L1:
  %phi = phi i32 [ 1, %L2 ], [ 2, %entry ]
  ret i32 %phi

L2:
  br label %L1
}

; RESULT: define i32 @foo_addrspacecast(
define i32 @foo_addrspacecast(i64 %arg0) {
entry:
  %gep = getelementptr inbounds [2 x ptr addrspace(1)], ptr @blockaddr.table.addrspacecast, i64 0, i64 %arg0
  %load = load ptr addrspace(1), ptr %gep, align 8
  indirectbr ptr addrspace(1) %load, [label %L2, label %L1]

L1:
  %phi = phi i32 [ 1, %L2 ], [ 2, %entry ]
  ret i32 %phi

L2:
  br label %L1
}
