; RUN: llc -stop-after=machine-sink < %s | FileCheck %s
;
; test code:
;  1 extern int bar(int x);
;  2
;  3 int foo(int *begin, int *end) {
;  4   int *i;
;  5   int ret = 0;
;  6   for (
;  7       i = begin ;
;  8       i != end ;
;  9       i++)
; 10   {
; 11       ret += bar(*i);
; 12   }
; 13   return ret;
; 14 }
;
; With the test code, LLVM-IR below shows that loop-control branches have a
; debug location of line 6 (branches in entry and for.body block). Make sure that
; these debug locations are propaged correctly to lowered instructions.
;
; CHECK: [[DLOC:![0-9]+]] = !DILocation(line: 6
; CHECK-DAG: [[VREG1:%[^ ]+]]:gr64 = COPY $rsi
; CHECK-DAG: [[VREG2:%[^ ]+]]:gr64 = COPY $rdi
; CHECK: SUB64rr [[VREG2]], [[VREG1]]
; CHECK-NEXT: JCC_1 {{.*}}, debug-location [[DLOC]]{{$}}
; CHECK: [[VREG3:%[^ ]+]]:gr64 = PHI [[VREG2]]
; CHECK: [[VREG4:%[^ ]+]]:gr64 = nuw ADD64ri8 [[VREG3]], 4
; CHECK: SUB64rr [[VREG4]], [[VREG1]]
; CHECK-NEXT: JCC_1 {{.*}}, debug-location [[DLOC]]{{$}}
; CHECK-NEXT: JMP_1 {{.*}}, debug-location [[DLOC]]{{$}}

target triple = "x86_64-unknown-linux-gnu"

define i32 @foo(i32* readonly %begin, i32* readnone %end) !dbg !4 {
entry:
  %cmp6 = icmp eq i32* %begin, %end, !dbg !9
  br i1 %cmp6, label %for.end, label %for.body.preheader, !dbg !12

for.body.preheader:                               ; preds = %entry
  br label %for.body, !dbg !13

for.body:                                         ; preds = %for.body.preheader, %for.body
  %ret.08 = phi i32 [ %add, %for.body ], [ 0, %for.body.preheader ]
  %i.07 = phi i32* [ %incdec.ptr, %for.body ], [ %begin, %for.body.preheader ]
  %0 = load i32, i32* %i.07, align 4, !dbg !13, !tbaa !15
  %call = tail call i32 @bar(i32 %0), !dbg !19
  %add = add nsw i32 %call, %ret.08, !dbg !20
  %incdec.ptr = getelementptr inbounds i32, i32* %i.07, i64 1, !dbg !21
  %cmp = icmp eq i32* %incdec.ptr, %end, !dbg !9
  br i1 %cmp, label %for.end.loopexit, label %for.body, !dbg !12, !llvm.loop !22

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end, !dbg !24

for.end:                                          ; preds = %for.end.loopexit, %entry
  %ret.0.lcssa = phi i32 [ 0, %entry ], [ %add, %for.end.loopexit ]
  ret i32 %ret.0.lcssa, !dbg !24
}

declare i32 @bar(i32)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1)
!1 = !DIFile(filename: "foo.c", directory: "b/")
!2 = !{i32 2, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 3, type: !5, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: true, unit: !0)
!5 = !DISubroutineType(types: !6)
!6 = !{!7, !8, !8}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64)
!9 = !DILocation(line: 8, column: 9, scope: !10)
!10 = distinct !DILexicalBlock(scope: !11, file: !1, line: 6, column: 3)
!11 = distinct !DILexicalBlock(scope: !4, file: !1, line: 6, column: 3)
!12 = !DILocation(line: 6, column: 3, scope: !11)
!13 = !DILocation(line: 11, column: 18, scope: !14)
!14 = distinct !DILexicalBlock(scope: !10, file: !1, line: 10, column: 3)
!15 = !{!16, !16, i64 0}
!16 = !{!"int", !17, i64 0}
!17 = !{!"omnipotent char", !18, i64 0}
!18 = !{!"Simple C/C++ TBAA"}
!19 = !DILocation(line: 11, column: 14, scope: !14)
!20 = !DILocation(line: 11, column: 11, scope: !14)
!21 = !DILocation(line: 9, column: 8, scope: !10)
!22 = distinct !{!22, !12, !23}
!23 = !DILocation(line: 12, column: 3, scope: !11)
!24 = !DILocation(line: 13, column: 3, scope: !4)
