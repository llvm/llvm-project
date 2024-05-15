; RUN: opt %loadPolly -polly-codegen -S < %s | FileCheck %s -check-prefix=SEQUENTIAL
; RUN: opt %loadPolly -polly-codegen -polly-ast-detect-parallel -S < %s | FileCheck %s -check-prefix=PARALLEL
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

; This is a trivially parallel loop. We just use it to ensure that we actually
; emit the right information.
;
; for (i = 0; i < n; i++)
;   A[i] = 1;
;
@A = common global [1024 x i32] zeroinitializer
define void @test-one(i64 %n) {
start:
  fence seq_cst
  br label %loop.header

loop.header:
  %i = phi i64 [ 0, %start ], [ %i.next, %loop.backedge ]
  %exitcond = icmp ne i64 %i, %n
  br i1 %exitcond, label %loop.body, label %ret

loop.body:
  %scevgep = getelementptr [1024 x i32], ptr @A, i64 0, i64 %i
  store i32 1, ptr %scevgep
  br label %loop.backedge

loop.backedge:
  %i.next = add nsw i64 %i, 1
  br label %loop.header

ret:
  fence seq_cst
  ret void
}

; SEQUENTIAL-LABEL: @test-one
; SEQUENTIAL-NOT: !llvm.mem.parallel_loop_access
; SEQUENTIAL-NOT: !llvm.access.group
; SEQUENTIAL-NOT: !llvm.loop

; PARALLEL: @test-one
; PARALLEL: store i32 1, ptr %scevgep1, {{[ ._!,a-zA-Z0-9]*}}, !llvm.access.group ![[GROUPID3:[0-9]+]]
; PARALLEL:  br i1 %polly.loop_cond, label %polly.loop_header, label %polly.loop_exit, !llvm.loop ![[LoopID4:[0-9]+]]


; This loop has memory dependences that require at least a simple dependence
; analysis to detect the parallelism.
;
; for (i = 0; i < n; i++)
;   A[2 * i] = A[2 * i + 1];
;
define void @test-two(i64 %n) {
start:
  fence seq_cst
  br label %loop.header

loop.header:
  %i = phi i64 [ 0, %start ], [ %i.next, %loop.backedge ]
  %exitcond = icmp ne i64 %i, %n
  br i1 %exitcond, label %loop.body, label %ret

loop.body:
  %loadoffset1 = mul nsw i64 %i, 2
  %loadoffset2 = add nsw i64 %loadoffset1, 1
  %scevgepload = getelementptr [1024 x i32], ptr @A, i64 0, i64 %loadoffset2
  %val = load i32, ptr %scevgepload
  %storeoffset = mul i64 %i, 2
  %scevgepstore = getelementptr [1024 x i32], ptr @A, i64 0, i64 %storeoffset
  store i32 %val, ptr %scevgepstore
  br label %loop.backedge

loop.backedge:
  %i.next = add nsw i64 %i, 1
  br label %loop.header

ret:
  fence seq_cst
  ret void
}

; SEQUENTIAL-LABEL: @test-two
; SEQUENTIAL-NOT: !llvm.mem.parallel_loop_access
; SEQUENTIAL-NOT: !llvm.access.group
; SEQUENTIAL-NOT: !llvm.loop

; PARALLEL: @test-two
; PARALLEL: %val_p_scalar_ = load i32, ptr %scevgep, {{[ ._!,a-zA-Z0-9]*}}, !llvm.access.group ![[GROUPID8:[0-9]*]]
; PARALLEL: store i32 %val_p_scalar_, ptr %scevgep1, {{[ ._!,a-zA-Z0-9]*}}, !llvm.access.group ![[GROUPID8]]
; PARALLEL:  br i1 %polly.loop_cond, label %polly.loop_header, label %polly.loop_exit, !llvm.loop ![[LoopID9:[0-9]*]]


; PARALLEL: ![[LoopID4]] = distinct !{![[LoopID4]], ![[PARACC5:[0-9]+]]}
; PARALLEL: ![[PARACC5]] = !{!"llvm.loop.parallel_accesses", ![[GROUPID3]]}
; PARALLEL: ![[LoopID9]] = distinct !{![[LoopID9]], ![[PARACC10:[0-9]+]]}
; PARALLEL: ![[PARACC10]] = !{!"llvm.loop.parallel_accesses", ![[GROUPID8]]}
