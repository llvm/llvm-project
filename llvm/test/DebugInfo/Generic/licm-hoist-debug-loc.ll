; RUN: opt -S -licm %s | FileCheck %s
;
; LICM should null out debug locations when it hoists instructions out of a loop.
;
; Generated with
; clang -O0 -S -emit-llvm test.cpp -g -gline-tables-only -o t.ll
; opt -S -sroa -adce -simplifycfg -reassociate -domtree -loops \
;     -loop-simplify -lcssa -basic-aa -aa -scalar-evolution -loop-rotate t.ll > test.ll
;
; void bar(int *);
; void foo(int k, int p)
; {
;    for (int i = 0; i < k; i++) {
;      bar(&p + 4);
;    }
; }
;
; We make sure that the instruction that is hoisted into the preheader
; does not have a debug location.
; CHECK: for.body.lr.ph:
; CHECK: getelementptr{{.*}}%p.addr, i64 4{{$}}
; CHECK: for.body:
;
; ModuleID = 't.ll'
source_filename = "test.c"

; Function Attrs: nounwind sspstrong uwtable
define void @foo(i32 %k, i32 %p) !dbg !7 {
entry:
  %p.addr = alloca i32, align 4
  store i32 %p, i32* %p.addr, align 4
  %cmp2 = icmp slt i32 0, %k, !dbg !9
  br i1 %cmp2, label %for.body.lr.ph, label %for.end, !dbg !9

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body, !dbg !9

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %i.03 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %add.ptr = getelementptr inbounds i32, i32* %p.addr, i64 4, !dbg !11
  call void @bar(i32* %add.ptr), !dbg !11
  %inc = add nsw i32 %i.03, 1, !dbg !12
  %cmp = icmp slt i32 %inc, %k, !dbg !9
  br i1 %cmp, label %for.body, label %for.cond.for.end_crit_edge, !dbg !9, !llvm.loop !14

for.cond.for.end_crit_edge:                       ; preds = %for.body
  br label %for.end, !dbg !9

for.end:                                          ; preds = %for.cond.for.end_crit_edge, %entry
  ret void, !dbg !16
}

declare void @bar(i32*)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.9.1 (PS4 clang version 4.50.0.249 7e7cd823 checking)", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2)
!1 = !DIFile(filename: "test.c", directory: "D:\test")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"PIC Level", i32 2}
!6 = !{!"clang version 3.9.1 (PS4 clang version 4.50.0.249 7e7cd823 checking)"}
!7 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 2, type: !8, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !2)
!9 = !DILocation(line: 4, scope: !10)
!10 = !DILexicalBlockFile(scope: !7, file: !1, discriminator: 1)
!11 = !DILocation(line: 5, scope: !7)
!12 = !DILocation(line: 4, scope: !13)
!13 = !DILexicalBlockFile(scope: !7, file: !1, discriminator: 2)
!14 = distinct !{!14, !15}
!15 = !DILocation(line: 4, scope: !7)
!16 = !DILocation(line: 7, scope: !7)
