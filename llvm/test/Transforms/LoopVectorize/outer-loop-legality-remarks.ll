; RUN: opt -passes=loop-vectorize -enable-vplan-native-path -disable-output \
; RUN:   -pass-remarks-output=- %s | FileCheck %s --match-full-lines \
; RUN:   --implicit-check-not='--- !'

; CHECK:      --- !Analysis
; CHECK-NEXT: Pass:            loop-vectorize
; CHECK-NEXT: Name:            CantVectorizeInstructionReturnType
; CHECK-NEXT: DebugLoc:        { File: test.c, Line: 3, Column: 0 }
; CHECK-NEXT: Function:        vector_load_store
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          'loop not vectorized: '
; CHECK-NEXT:   - String:          instruction return type cannot be vectorized
; CHECK-NEXT: ...
; CHECK-NEXT: --- !Analysis
; CHECK-NEXT: Pass:            loop-vectorize
; CHECK-NEXT: Name:            CantVectorizeStore
; CHECK-NEXT: DebugLoc:        { File: test.c, Line: 4, Column: 0 }
; CHECK-NEXT: Function:        vector_load_store
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          'loop not vectorized: '
; CHECK-NEXT:   - String:          Store instruction cannot be vectorized
; CHECK-NEXT: ...
; CHECK-NEXT: --- !Analysis
; CHECK-NEXT: Pass:            loop-vectorize
; CHECK-NEXT: Name:            UnsupportedOuterLoop
; CHECK-NEXT: DebugLoc:        { File: test.c, Line: 2, Column: 0 }
; CHECK-NEXT: Function:        vector_load_store
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          'loop not vectorized: '
; CHECK-NEXT:   - String:          Unsupported outer loop
; CHECK-NEXT: ...
; CHECK-NEXT: --- !Missed
; CHECK-NEXT: Pass:            loop-vectorize
; CHECK-NEXT: Name:            MissedDetails
; CHECK-NEXT: DebugLoc:        { File: test.c, Line: 2, Column: 0 }
; CHECK-NEXT: Function:        vector_load_store
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          loop not vectorized
; CHECK-NEXT:   - String:          ' (Force='
; CHECK-NEXT:   - Force:           'true'
; CHECK-NEXT:   - String:          ')'
; CHECK-NEXT: ...
define void @vector_load_store(ptr noalias %A, ptr noalias %B, i64 %N, i64 %M) !dbg !3 {
entry:
  br label %outer.header

outer.header:
  %outer.iv = phi i64 [ 0, %entry ], [ %outer.iv.next, %outer.latch ]
  br label %inner

inner:
  %inner.iv = phi i64 [ 0, %outer.header ], [ %inner.iv.next, %inner ]
  %inner.iv.next = add nuw nsw i64 %inner.iv, 1
  %inner.ec = icmp eq i64 %inner.iv.next, %M
  br i1 %inner.ec, label %outer.latch, label %inner

outer.latch:
  %gep.B = getelementptr inbounds <2 x i32>, ptr %B, i64 %outer.iv
  %l = load <2 x i32>, ptr %gep.B, align 8, !dbg !7
  %gep.A = getelementptr inbounds <2 x i32>, ptr %A, i64 %outer.iv
  store <2 x i32> %l, ptr %gep.A, align 8, !dbg !8
  %outer.iv.next = add nuw nsw i64 %outer.iv, 1
  %outer.ec = icmp eq i64 %outer.iv.next, %N
  br i1 %outer.ec, label %exit, label %outer.header, !llvm.loop !4

exit:
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "test.c", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "vector_load_store", scope: !1, file: !1, line: 1, type: !9, spFlags: DISPFlagDefinition, unit: !0)
!4 = distinct !{!4, !5, !6}
!5 = !DILocation(line: 2, scope: !3)
!6 = !{!"llvm.loop.vectorize.enable"}
!7 = !DILocation(line: 3, scope: !3)
!8 = !DILocation(line: 4, scope: !3)
!9 = !DISubroutineType(types: !{})
