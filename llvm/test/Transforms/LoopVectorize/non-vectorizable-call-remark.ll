; RUN: opt < %s -passes=loop-vectorize -S -pass-remarks-analysis='loop-vectorize' -disable-output 2>&1 | FileCheck %s
; RUN: opt < %s -passes=loop-vectorize -o /dev/null -pass-remarks-output=%t.yaml
; RUN: cat %t.yaml | FileCheck -check-prefix=YAML %s

; Verify that when a call instruction in a loop cannot be vectorized,
; the optimization remark points to the call instruction's debug location
; (source.c:7:10), not the loop header (source.c:5:3).
;
; C source code for the tests (line numbers match debug metadata):
;
;  1: int opaque(int *);
;  2: void opaque_write(int *);
;  3:
;  4: void call_reads_memory(int *p, long n) {
;  5:   for (long i = 0; i < n; i++) {
;  6:     int *addr = &p[i];
;  7:     int val = opaque(addr);
;  8:     p[i] = val;
;  9:   }
; 10: }
; 11:
; 12:
; 13:
; 14: void call_writes_memory(int *p, long n) {
; 15:   for (long i = 0; i < n; i++) {
; 16:     int *addr = &p[i];
; 17:     opaque_write(addr);
; 18:   }
; 19:
; 20: }

; File, line, and column should match those specified in the metadata.
; The remark should point at the call instruction, not the loop header.
; CHECK: remark: source.c:7:10: loop not vectorized: instruction cannot be vectorized
; CHECK: remark: source.c:17:5: loop not vectorized: instruction cannot be vectorized

; YAML:      --- !Analysis
; YAML:      Pass:            loop-vectorize
; YAML:      Name:            CantVectorizeInstruction
; YAML-NEXT: DebugLoc:        { File: source.c, Line: 7, Column: 10 }
; YAML-NEXT: Function:        call_reads_memory
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'loop not vectorized: '
; YAML-NEXT:   - String:          instruction cannot be vectorized
; YAML:      --- !Analysis
; YAML:      Pass:            loop-vectorize
; YAML:      Name:            CantVectorizeInstruction
; YAML-NEXT: DebugLoc:        { File: source.c, Line: 17, Column: 5 }
; YAML-NEXT: Function:        call_writes_memory
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'loop not vectorized: '
; YAML-NEXT:   - String:          instruction cannot be vectorized

define void @call_reads_memory(ptr %p, i64 %n) !dbg !7 {
entry:
  br label %loop, !dbg !9

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %gep = getelementptr inbounds i32, ptr %p, i64 %iv, !dbg !10
  %val = call i32 @opaque(ptr %gep), !dbg !11
  store i32 %val, ptr %gep, align 4, !dbg !12
  %iv.next = add nuw nsw i64 %iv, 1, !dbg !13
  %exitcond = icmp eq i64 %iv.next, %n, !dbg !14
  br i1 %exitcond, label %exit, label %loop, !dbg !15

exit:
  ret void, !dbg !16
}

define void @call_writes_memory(ptr %p, i64 %n) !dbg !17 {
entry:
  br label %loop, !dbg !19

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %gep = getelementptr inbounds i32, ptr %p, i64 %iv, !dbg !20
  call void @opaque_write(ptr %gep), !dbg !21
  %iv.next = add nuw nsw i64 %iv, 1, !dbg !22
  %exitcond = icmp eq i64 %iv.next, %n, !dbg !23
  br i1 %exitcond, label %exit, label %loop, !dbg !24

exit:
  ret void, !dbg !25
}

declare i32 @opaque(ptr)
declare void @opaque_write(ptr)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "source.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !DISubroutineType(types: !2)
!6 = !DILocation(line: 5, column: 3, scope: !7)
!7 = distinct !DISubprogram(name: "call_reads_memory", scope: !1, file: !1, line: 4, type: !5, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!8 = !DILocation(line: 5, column: 3, scope: !7)
!9 = !DILocation(line: 5, column: 3, scope: !7)
!10 = !DILocation(line: 6, column: 5, scope: !7)
!11 = !DILocation(line: 7, column: 10, scope: !7)
!12 = !DILocation(line: 8, column: 5, scope: !7)
!13 = !DILocation(line: 5, column: 28, scope: !7)
!14 = !DILocation(line: 5, column: 20, scope: !7)
!15 = !DILocation(line: 5, column: 3, scope: !7)
!16 = !DILocation(line: 10, column: 1, scope: !7)
!17 = distinct !DISubprogram(name: "call_writes_memory", scope: !1, file: !1, line: 14, type: !5, scopeLine: 14, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!18 = !DILocation(line: 15, column: 3, scope: !17)
!19 = !DILocation(line: 15, column: 3, scope: !17)
!20 = !DILocation(line: 16, column: 5, scope: !17)
!21 = !DILocation(line: 17, column: 5, scope: !17)
!22 = !DILocation(line: 15, column: 28, scope: !17)
!23 = !DILocation(line: 15, column: 20, scope: !17)
!24 = !DILocation(line: 15, column: 3, scope: !17)
!25 = !DILocation(line: 20, column: 1, scope: !17)
