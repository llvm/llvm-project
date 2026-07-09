; RUN: opt < %s -passes=loop-vectorize -disable-output -pass-remarks-output=- 2>&1 | FileCheck %s

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

; The remark should point at the call instruction, not the loop header.
; CHECK:      --- !Analysis
; CHECK:      Pass:            loop-vectorize
; CHECK:      Name:            CantVectorizeInstruction
; CHECK-NEXT: DebugLoc:        { File: source.c, Line: 7, Column: 10 }
; CHECK-NEXT: Function:        call_reads_memory
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          'loop not vectorized: '
; CHECK-NEXT:   - String:          instruction cannot be vectorized
; CHECK:      --- !Analysis
; CHECK:      Pass:            loop-vectorize
; CHECK:      Name:            CantVectorizeInstruction
; CHECK-NEXT: DebugLoc:        { File: source.c, Line: 17, Column: 5 }
; CHECK-NEXT: Function:        call_writes_memory
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          'loop not vectorized: '
; CHECK-NEXT:   - String:          instruction cannot be vectorized

define void @call_reads_memory(ptr %p, i64 %n) !dbg !4 {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %gep = getelementptr inbounds i32, ptr %p, i64 %iv
  %val = call i32 @opaque(ptr %gep), !dbg !5
  store i32 %val, ptr %gep, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

define void @call_writes_memory(ptr %p, i64 %n) !dbg !6 {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %gep = getelementptr inbounds i32, ptr %p, i64 %iv
  call void @opaque_write(ptr %gep), !dbg !7
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

declare i32 @opaque(ptr)
declare void @opaque_write(ptr)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "source.c", directory: "/tmp")
!2 = !{i32 2, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "call_reads_memory", scope: !1, file: !1, line: 4, type: !8, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!5 = !DILocation(line: 7, column: 10, scope: !4)
!6 = distinct !DISubprogram(name: "call_writes_memory", scope: !1, file: !1, line: 14, type: !8, scopeLine: 14, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!7 = !DILocation(line: 17, column: 5, scope: !6)
!8 = !DISubroutineType(types: !{})
