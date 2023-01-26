; RUN: opt -passes='function(loop-vectorize,require<access-info>)' -disable-output -pass-remarks-analysis=loop-vectorize < %s 2>&1 | FileCheck %s
; RUN: opt < %s -passes='function(require<access-info>,loop-vectorize)' -o /dev/null -pass-remarks-output=%t.yaml
; RUN: cat %t.yaml | FileCheck -check-prefix=YAML %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"

; // Loop has an array element B[i] (%arrayidx in IR) being used as index to
; // another array (A), and since the value of B[i] is unknown,
; // the bound for array A is unknown.
; void test_unknown_bounds(int n, int* A, int* B) {
;     for(int i = 0; i < n ; ++i)
;         A[i] = A[B[i]] + 1;
; }

; CHECK: remark: source.c:4:16: loop not vectorized: cannot identify array bounds

define void @test_unknown_bounds(i64 %n, ptr nocapture %A, ptr nocapture readonly %B) !dbg !13 {
entry:
  %cmp10 = icmp sgt i64 %n, 0
  br i1 %cmp10, label %for.body, label %for.cond.cleanup

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %B, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %idxprom1 = sext i32 %0 to i64, !dbg !35
  %arrayidx2 = getelementptr inbounds i32, ptr %A, i64 %idxprom1, !dbg !35
  %1 = load i32, ptr %arrayidx2, align 4, !dbg !35
  %add = add nsw i32 %1, 1
  %arrayidx4 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  store i32 %add, ptr %arrayidx4, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body, !dbg !28

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void
}

; // a) Dependence::NoDep
; // Loop containing only reads (here of the array A) does not hinder vectorization
; void test_nodep(int n, int* A, int* B) {
;   for(int i = 1; i < n ; ++i) {
;     B[i] = A[i-1] + A[i+2];
;   }
; }

; CHECK-NOT: remark: source.c:{{0-9]+}}:{{[0-9]+}}:

define void @test_nodep(i64 %n, ptr nocapture readonly %A, ptr nocapture %B) !dbg !44 {
entry:
  %cmp12 = icmp sgt i64 %n, 1
  br i1 %cmp12, label %for.body, label %for.cond.cleanup

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 1, %entry ], [ %indvars.iv.next, %for.body ]
  %0 = add nsw i64 %indvars.iv, -1
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %0, !dbg !61
  %1 = load i32, ptr %arrayidx, align 4, !dbg !61
  %2 = add nuw nsw i64 %indvars.iv, 2
  %arrayidx2 = getelementptr inbounds i32, ptr %A, i64 %2, !dbg !63
  %3 = load i32, ptr %arrayidx2, align 4, !dbg !63
  %add3 = add nsw i32 %3, %1
  %arrayidx5 = getelementptr inbounds i32, ptr %B, i64 %indvars.iv
  store i32 %add3, ptr %arrayidx5, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void
}


; // b) Dependence::Forward
; // Loop gets vectorized since it contains only a forward
; // dependency between A[i-2] and A[i]
; void test_forward(int n, int* A, int* B) {
;   for(int i=1; i < n; ++i) {
;     A[i] = 10;
;     B[i] = A[i-2];
;   }
; }

; CHECK-NOT: remark: source.c:{{0-9]+}}:{{[0-9]+}}:
define dso_local void @test_forward(i64 %n, ptr nocapture %A, ptr nocapture %B) !dbg !70 {
entry:
  %cmp11 = icmp sgt i64 %n, 1
  br i1 %cmp11, label %for.body, label %for.cond.cleanup, !dbg !81

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 1, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %indvars.iv, !dbg !83
  store i32 10, ptr %arrayidx, align 4
  %0 = add nsw i64 %indvars.iv, -2
  %arrayidx2 = getelementptr inbounds i32, ptr %A, i64 %0, !dbg !87
  %1 = load i32, ptr %arrayidx2, align 4, !dbg !87
  %arrayidx4 = getelementptr inbounds i32, ptr %B, i64 %indvars.iv, !dbg !88
  store i32 %1, ptr %arrayidx4, align 4, !dbg !89
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body, !dbg !81

  for.cond.cleanup:                                 ; preds = %for.body, %entry
    ret void
}


; // c) Dependence::BackwardVectorizable
; // Loop gets vectorized since it contains a backward dependency
; // between A[i] and A[i-4], but the dependency distance (4) is
; // greater than the minimum possible VF (2 in this case)
; void test_backwardVectorizable(int n, int* A) {
;   for(int i=4; i < n; ++i) {
;     A[i] = A[i-4] + 1;
;   }
; }

; CHECK-NOT: remark: source.c:{{0-9]+}}:{{[0-9]+}}:

define dso_local void @test_backwardVectorizable(i64 %n, ptr nocapture %A) !dbg !93 {
entry:
  %cmp8 = icmp sgt i64 %n, 4
  br i1 %cmp8, label %for.body, label %for.cond.cleanup

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 4, %entry ], [ %indvars.iv.next, %for.body ]
  %0 = add nsw i64 %indvars.iv, -4, !dbg !106
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %0, !dbg !108
  %1 = load i32, ptr %arrayidx, align 4, !dbg !108
  %add = add nsw i32 %1, 1
  %arrayidx2 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv, !dbg !110
  store i32 %add, ptr %arrayidx2, align 4, !dbg !111
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

  for.cond.cleanup:                                 ; preds = %for.body, %entry
    ret void
}

; // d) Dependence::Backward
; // Loop does not get vectorized since it contains a backward
; // dependency between A[i] and A[i+3].
; void test_backward_dep(int n, int *A) {
;   for (int i = 1; i <= n - 3; i += 3) {
;     A[i] = A[i-1];
;     A[i+1] = A[i+3];
;   }
; }

; CHECK: remark: source.c:48:14: loop not vectorized: unsafe dependent memory operations in loop. Use #pragma loop distribute(enable) to allow loop distribution to attempt to isolate the offending operations into a separate loop
; CHECK-NEXT: Backward loop carried data dependence. Memory location is the same as accessed at source.c:47:5

define void @test_backward_dep(i64 %n, ptr nocapture %A) {
entry:
  %cmp.not19 = icmp slt i64 %n, 4
  br i1 %cmp.not19, label %for.cond.cleanup, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  %sub = add nsw i64 %n, -3
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 1, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %0 = add nsw i64 %indvars.iv, -1
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %0
  %1 = load i32, ptr %arrayidx, align 8
  %arrayidx3 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv, !dbg !157
  store i32 %1, ptr %arrayidx3, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 3
  %arrayidx5 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv.next, !dbg !160
  %2 = load i32, ptr %arrayidx5, align 8, !dbg !160
  %3 = add nuw nsw i64 %indvars.iv, 1
  %arrayidx8 = getelementptr inbounds i32, ptr %A, i64 %3
  store i32 %2, ptr %arrayidx8, align 8
  %cmp.not = icmp ugt i64 %indvars.iv.next, %n
  br i1 %cmp.not, label %for.cond.cleanup, label %for.body

  for.cond.cleanup:                                 ; preds = %for.body, %entry
    ret void
}

; // e) Dependence::ForwardButPreventsForwarding
; // Loop does not get vectorized despite only having a forward
; // dependency between A[i] and A[i-3].
; // This is because the store-to-load forwarding distance (here 3)
; // needs to be a multiple of vector factor otherwise the
; // store (A[5:6] in i=5) and load (A[4:5],A[6:7] in i=7,9) are unaligned.
; void test_forwardButPreventsForwarding_dep(int n, int* A, int* B) {
;   for(int i=3; i < n; ++i) {
;     A[i] = 10;
;     B[i] = A[i-3];
;   }
; }

; CHECK: remark: source.c:61:12: loop not vectorized: unsafe dependent memory operations in loop. Use #pragma loop distribute(enable) to allow loop distribution to attempt to isolate the offending operations into a separate loop
; CHECK-NEXT: Forward loop carried data dependence that prevents store-to-load forwarding. Memory location is the same as accessed at source.c:60:5

define void @test_forwardButPreventsForwarding_dep(i64 %n, ptr nocapture %A, ptr nocapture %B) !dbg !166 {
entry:
  %cmp11 = icmp sgt i64 %n, 3
  br i1 %cmp11, label %for.body, label %for.cond.cleanup

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 3, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %indvars.iv, !dbg !179
  store i32 10, ptr %arrayidx, align 4
  %0 = add nsw i64 %indvars.iv, -3
  %arrayidx2 = getelementptr inbounds i32, ptr %A, i64 %0, !dbg !183
  %1 = load i32, ptr %arrayidx2, align 4, !dbg !183
  %arrayidx4 = getelementptr inbounds i32, ptr %B, i64 %indvars.iv
  store i32 %1, ptr %arrayidx4, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

  for.cond.cleanup:                                 ; preds = %for.body, %entry
    ret void
}

; // f) Dependence::BackwardVectorizableButPreventsForwarding
; // Loop does not get vectorized despite having a backward
; // but vectorizable dependency between A[i] and A[i-15].
; //
; // This is because the store-to-load forwarding distance (here 15)
; // needs to be a multiple of vector factor otherwise
; // store (A[16:17] in i=16) and load (A[15:16], A[17:18] in i=30,32) are unaligned.
; void test_backwardVectorizableButPreventsForwarding(int n, int* A) {
;   for(int i=15; i < n; ++i) {
;     A[i] = A[i-2] + A[i-15];
;   }
; }

; CHECK: remark: source.c:74:5: loop not vectorized: unsafe dependent memory operations in loop. Use #pragma loop distribute(enable) to allow loop distribution to attempt to isolate the offending operations into a separate loop
; CHECK: Backward loop carried data dependence that prevents store-to-load forwarding. Memory location is the same as accessed at source.c:74:21

define void @test_backwardVectorizableButPreventsForwarding(i64 %n, ptr nocapture %A) !dbg !189 {
entry:
  %cmp13 = icmp sgt i64 %n, 15
  br i1 %cmp13, label %for.body, label %for.cond.cleanup

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 15, %entry ], [ %indvars.iv.next, %for.body ]
  %0 = add nsw i64 %indvars.iv, -2
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %0
  %1 = load i32, ptr %arrayidx, align 4
  %2 = add nsw i64 %indvars.iv, -15
  %arrayidx3 = getelementptr inbounds i32, ptr %A, i64 %2, !dbg !207
  %3 = load i32, ptr %arrayidx3, align 4
  %add = add nsw i32 %3, %1
  %arrayidx5 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv, !dbg !209
  store i32 %add, ptr %arrayidx5, align 4, !dbg !209
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

  for.cond.cleanup:                                 ; preds = %for.body, %entry
    ret void
}

; // g) Dependence::Unknown
; // Different stride lengths
; void test_unknown_dep(int n, int* A) {
;   for(int i=0; i < n; ++i) {
;       A[(i+1)*4] = 10;
;       A[i] = 100;
;   }
; }

; CHECK: remark: source.c:83:7: loop not vectorized: unsafe dependent memory operations in loop. Use #pragma loop distribute(enable) to allow loop distribution to attempt to isolate the offending operations into a separate loop
; CHECK: Unknown data dependence. Memory location is the same as accessed at source.c:82:7

define void @test_unknown_dep(i64 %n, ptr nocapture %A) !dbg !214 {
entry:
  %cmp8 = icmp sgt i64 %n, 0
  br i1 %cmp8, label %for.body, label %for.cond.cleanup

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %0 = shl nsw i64 %indvars.iv.next, 2
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %0, !dbg !229
  store i32 10, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv, !dbg !231
  store i32 100, ptr %arrayidx2, align 4, !dbg !231
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

  for.cond.cleanup:                                 ; preds = %for.body, %entry
    ret void
}

; YAML:      --- !Analysis
; YAML-NEXT: Pass:            loop-vectorize
; YAML-NEXT: Name:            CantIdentifyArrayBounds
; YAML-NEXT: DebugLoc:        { File: source.c, Line: 4, Column: 16 }
; YAML-NEXT: Function:        test_unknown_bounds
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'loop not vectorized: '
; YAML-NEXT:   - String:          cannot identify array bounds
; YAML-NEXT: ...
; YAML-NEXT: --- !Missed
; YAML-NEXT: Pass:            loop-vectorize
; YAML-NEXT: Name:            MissedDetails
; YAML-NEXT: DebugLoc:        { File: source.c, Line: 3, Column: 5 }
; YAML-NEXT: Function:        test_unknown_bounds
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          loop not vectorized
; YAML-NEXT: ...
; YAML:      --- !Analysis
; YAML-NEXT: Pass:            loop-vectorize
; YAML-NEXT: Name:            UnsafeDep
; YAML-NEXT: DebugLoc:        { File: source.c, Line: 48, Column: 14 }
; YAML-NEXT: Function:        test_backward_dep
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'loop not vectorized: '
; YAML-NEXT:   - String:          'unsafe dependent memory operations in loop. Use #pragma loop distribute(enable) to allow loop distribution to attempt to isolate the offending operations into a separate loop'
; YAML-NEXT:   - String:          "\nBackward loop carried data dependence."
; YAML-NEXT:   - String:          ' Memory location is the same as accessed at '
; YAML-NEXT:   - Location:        'source.c:47:5'
; YAML-NEXT:     DebugLoc:        { File: source.c, Line: 47, Column: 5 }
; YAML-NEXT: ...
; YAML-NEXT: --- !Missed
; YAML-NEXT: Pass:            loop-vectorize
; YAML-NEXT: Name:            MissedDetails
; YAML-NEXT: Function:        test_backward_dep
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          loop not vectorized
; YAML-NEXT: ...
; YAML-NEXT: --- !Analysis
; YAML-NEXT: Pass:            loop-vectorize
; YAML-NEXT: Name:            UnsafeDep
; YAML-NEXT: DebugLoc:        { File: source.c, Line: 61, Column: 12 }
; YAML-NEXT: Function:        test_forwardButPreventsForwarding_dep
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'loop not vectorized: '
; YAML-NEXT:   - String:          'unsafe dependent memory operations in loop. Use #pragma loop distribute(enable) to allow loop distribution to attempt to isolate the offending operations into a separate loop'
; YAML-NEXT:   - String:          "\nForward loop carried data dependence that prevents store-to-load forwarding."
; YAML-NEXT:   - String:          ' Memory location is the same as accessed at '
; YAML-NEXT:   - Location:        'source.c:60:5'
; YAML-NEXT:     DebugLoc:        { File: source.c, Line: 60, Column: 5 }
; YAML-NEXT: ...
; YAML-NEXT: --- !Missed
; YAML-NEXT: Pass:            loop-vectorize
; YAML-NEXT: Name:            MissedDetails
; YAML-NEXT: Function:        test_forwardButPreventsForwarding_dep
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          loop not vectorized
; YAML-NEXT: ...
; YAML-NEXT: --- !Analysis
; YAML-NEXT: Pass:            loop-vectorize
; YAML-NEXT: Name:            UnsafeDep
; YAML-NEXT: DebugLoc:        { File: source.c, Line: 74, Column: 5 }
; YAML-NEXT: Function:        test_backwardVectorizableButPreventsForwarding
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'loop not vectorized: '
; YAML-NEXT:   - String:          'unsafe dependent memory operations in loop. Use #pragma loop distribute(enable) to allow loop distribution to attempt to isolate the offending operations into a separate loop'
; YAML-NEXT:   - String:          "\nBackward loop carried data dependence that prevents store-to-load forwarding."
; YAML-NEXT:   - String:          ' Memory location is the same as accessed at '
; YAML-NEXT:   - Location:        'source.c:74:21'
; YAML-NEXT:     DebugLoc:        { File: source.c, Line: 74, Column: 21 }
; YAML-NEXT: ...
; YAML-NEXT: --- !Missed
; YAML-NEXT: Pass:            loop-vectorize
; YAML-NEXT: Name:            MissedDetails
; YAML-NEXT: Function:        test_backwardVectorizableButPreventsForwarding
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          loop not vectorized
; YAML-NEXT: ...
; YAML-NEXT: --- !Analysis
; YAML-NEXT: Pass:            loop-vectorize
; YAML-NEXT: Name:            UnsafeDep
; YAML-NEXT: DebugLoc:        { File: source.c, Line: 83, Column: 7 }
; YAML-NEXT: Function:        test_unknown_dep
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'loop not vectorized: '
; YAML-NEXT:   - String:          'unsafe dependent memory operations in loop. Use #pragma loop distribute(enable) to allow loop distribution to attempt to isolate the offending operations into a separate loop'
; YAML-NEXT:   - String:          "\nUnknown data dependence."
; YAML-NEXT:   - String:          ' Memory location is the same as accessed at '
; YAML-NEXT:   - Location:        'source.c:82:7'
; YAML-NEXT:     DebugLoc:        { File: source.c, Line: 82, Column: 7 }
; YAML-NEXT: ...
; YAML-NEXT: --- !Missed
; YAML-NEXT: Pass:            loop-vectorize
; YAML-NEXT: Name:            MissedDetails
; YAML-NEXT: Function:        test_unknown_dep
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          loop not vectorized
; YAML-NEXT: ...


!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 14.0.0 (https://github.com/llvm/llvm-project.git 54f0f826c5c7d0ff16c230b259cb6aad33e18d97)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "source.c", directory: "")
!2 = !{}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!13 = distinct !DISubprogram(name: "test_unknown_bounds", scope: !1, file: !1, line: 2, type: !45, scopeLine: 2, unit: !0, retainedNodes: !2)
!23 = distinct !DILexicalBlock(scope: !13, file: !1, line: 3, column: 5)
!27 = distinct !DILexicalBlock(scope: !23, file: !1, line: 3, column: 5)
!28 = !DILocation(line: 3, column: 5, scope: !23)
!35 = !DILocation(line: 4, column: 16, scope: !27)
!44 = distinct !DISubprogram(name: "test_nodep", scope: !1, file: !1, line: 14, type: !45, scopeLine: 14, unit: !0, retainedNodes: !2)
!45 = !DISubroutineType(types: !46)
!46 = !{null, !18, !16, !16}
!16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !17, size: 64)
!17 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!18 = !DIBasicType(name: "int", size: 64, encoding: DW_ATE_signed)
!52 = distinct !DILexicalBlock(scope: !44, file: !1, line: 15, column: 3)
!56 = distinct !DILexicalBlock(scope: !52, file: !1, line: 15, column: 3)
!60 = distinct !DILexicalBlock(scope: !56, file: !1, line: 15, column: 31)
!61 = !DILocation(line: 16, column: 12, scope: !60)
!63 = !DILocation(line: 16, column: 21, scope: !60)
!70 = distinct !DISubprogram(name: "test_forward", scope: !1, file: !1, line: 24, type: !45, scopeLine: 24, unit: !0, retainedNodes: !2)
!77 = distinct !DILexicalBlock(scope: !70, file: !1, line: 25, column: 3)
!80 = distinct !DILexicalBlock(scope: !77, file: !1, line: 25, column: 3)
!81 = !DILocation(line: 25, column: 3, scope: !77)
!83 = !DILocation(line: 26, column: 5, scope: !84)
!84 = distinct !DILexicalBlock(scope: !80, file: !1, line: 25, column: 28)
!87 = !DILocation(line: 27, column: 12, scope: !84)
!88 = !DILocation(line: 27, column: 5, scope: !84)
!89 = !DILocation(line: 27, column: 10, scope: !84)
!93 = distinct !DISubprogram(name: "test_backwardVectorizable", scope: !1, file: !1, line: 36, type: !95, scopeLine: 36, unit: !0, retainedNodes: !2)
!95 = !DISubroutineType(types: !96)
!96 = !{null, !18, !16}
!99 = distinct !DILexicalBlock(scope: !93, file: !1, line: 37, column: 3)
!103 = distinct !DILexicalBlock(scope: !99, file: !1, line: 37, column: 3)
!106 = !DILocation(line: 38, column: 15, scope: !107)
!107 = distinct !DILexicalBlock(scope: !103, file: !1, line: 37, column: 28)
!108 = !DILocation(line: 38, column: 12, scope: !107)
!110 = !DILocation(line: 38, column: 5, scope: !107)
!111 = !DILocation(line: 38, column: 10, scope: !107)
!136 = distinct !DISubprogram(name: "test_backward_dep", scope: !1, file: !1, line: 45, type: !95, scopeLine: 45, unit: !0, retainedNodes: !2)
!145 = distinct !DILexicalBlock(scope: !136, file: !1, line: 46, column: 3)
!149 = distinct !DILexicalBlock(scope: !145, file: !1, line: 46, column: 3)
!153 = distinct !DILexicalBlock(scope: !149, file: !1, line: 46, column: 39)
!157 = !DILocation(line: 47, column: 5, scope: !153)
!160 = !DILocation(line: 48, column: 14, scope: !153)
!166 = distinct !DISubprogram(name: "test_forwardButPreventsForwarding_dep", scope: !1, file: !1, line: 58, type: !45, scopeLine: 58, unit: !0, retainedNodes: !2)
!172 = distinct !DILexicalBlock(scope: !166, file: !1, line: 59, column: 3)
!176 = distinct !DILexicalBlock(scope: !172, file: !1, line: 59, column: 3)
!179 = !DILocation(line: 60, column: 5, scope: !180)
!180 = distinct !DILexicalBlock(scope: !176, file: !1, line: 59, column: 28)
!183 = !DILocation(line: 61, column: 12, scope: !180)
!189 = distinct !DISubprogram(name: "test_backwardVectorizableButPreventsForwarding", scope: !1, file: !1, line: 72, type: !95, scopeLine: 72, unit: !0, retainedNodes: !2)
!196 = distinct !DILexicalBlock(scope: !189, file: !1, line: 73, column: 3)
!200 = distinct !DILexicalBlock(scope: !196, file: !1, line: 73, column: 3)
!204 = distinct !DILexicalBlock(scope: !200, file: !1, line: 73, column: 29)
!207 = !DILocation(line: 74, column: 21, scope: !204)
!209 = !DILocation(line: 74, column: 5, scope: !204)
!214 = distinct !DISubprogram(name: "test_unknown_dep", scope: !1, file: !1, line: 80, type: !95, scopeLine: 80, unit: !0, retainedNodes: !2)
!219 = distinct !DILexicalBlock(scope: !214, file: !1, line: 81, column: 3)
!223 = distinct !DILexicalBlock(scope: !219, file: !1, line: 81, column: 3)
!227 = distinct !DILexicalBlock(scope: !223, file: !1, line: 81, column: 28)
!229 = !DILocation(line: 82, column: 7, scope: !227)
!231 = !DILocation(line: 83, column: 7, scope: !227)
