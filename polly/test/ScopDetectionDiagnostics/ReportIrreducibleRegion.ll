; RUN: opt %loadPolly -analyze -polly-detect \
; RUN:     -pass-remarks-missed="polly-detect" \
; RUN:     < %s 2>&1| FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

;void foo(int a, int b) {
;  if(b == 42) {
;    if (a > 0) {
;      LABEL1:
;      a--;
;    }
;
;    if (a > 0) {
;      goto LABEL1;
;    }
;    b = b + 42;
;  }
;}

; CHECK: remark: ReportIrreducibleRegion.c:3:7: The following errors keep this region from being a Scop.
; CHECK-NEXT: remark: ReportIrreducibleRegion.c:9:4: Irreducible region encountered in control flow.
; CHECK-NEXT: remark: ReportIrreducibleRegion.c:9:4: Invalid Scop candidate ends here.


; Function Attrs: nounwind uwtable
define void @foo(i32 %a, i32 %b) #0 !dbg !4 {
entry:
  %a.addr = alloca i32, align 4
  %b.addr = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %a.addr, metadata !11, metadata !12), !dbg !13
  store i32 %b, i32* %b.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %b.addr, metadata !14, metadata !12), !dbg !15
  %0 = load i32, i32* %b.addr, align 4, !dbg !16
  %cmp = icmp eq i32 %0, 42, !dbg !18
  br i1 %cmp, label %if.then, label %if.end6, !dbg !19

if.then:                                          ; preds = %entry
  %1 = load i32, i32* %a.addr, align 4, !dbg !20
  %cmp1 = icmp sgt i32 %1, 0, !dbg !23
  br i1 %cmp1, label %if.then2, label %if.end, !dbg !24

if.then2:                                         ; preds = %if.then
  br label %LABEL1, !dbg !25

LABEL1:                                           ; preds = %if.then4, %if.then2
  %2 = load i32, i32* %a.addr, align 4, !dbg !27
  %dec = add nsw i32 %2, -1, !dbg !27
  store i32 %dec, i32* %a.addr, align 4, !dbg !27
  br label %if.end, !dbg !29

if.end:                                           ; preds = %LABEL1, %if.then
  %3 = load i32, i32* %a.addr, align 4, !dbg !30
  %cmp3 = icmp sgt i32 %3, 0, !dbg !32
  br i1 %cmp3, label %if.then4, label %if.end5, !dbg !33

if.then4:                                         ; preds = %if.end
  br label %LABEL1, !dbg !34

if.end5:                                          ; preds = %if.end
  %4 = load i32, i32* %b.addr, align 4, !dbg !36
  %add = add nsw i32 %4, 42, !dbg !37
  store i32 %add, i32* %b.addr, align 4, !dbg !38
  br label %if.end6, !dbg !39

if.end6:                                          ; preds = %if.end5, %entry
  ret void, !dbg !40
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9}
!llvm.ident = !{!10}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.8.0", isOptimized: false, runtimeVersion: 0, emissionKind: 1, enums: !2)
!1 = !DIFile(filename: "ReportIrreducibleRegion.c", directory: "llvm/tools/polly/test/ScopDetectionDiagnostics")
!2 = !{}
!4 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !5, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{null, !7, !7}
!7 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = !{i32 2, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{!"clang version 3.8.0"}
!11 = !DILocalVariable(name: "a", arg: 1, scope: !4, file: !1, line: 1, type: !7)
!12 = !DIExpression()
!13 = !DILocation(line: 1, column: 14, scope: !4)
!14 = !DILocalVariable(name: "b", arg: 2, scope: !4, file: !1, line: 1, type: !7)
!15 = !DILocation(line: 1, column: 21, scope: !4)
!16 = !DILocation(line: 2, column: 5, scope: !17)
!17 = distinct !DILexicalBlock(scope: !4, file: !1, line: 2, column: 5)
!18 = !DILocation(line: 2, column: 7, scope: !17)
!19 = !DILocation(line: 2, column: 5, scope: !4)
!20 = !DILocation(line: 3, column: 7, scope: !21)
!21 = distinct !DILexicalBlock(scope: !22, file: !1, line: 3, column: 7)
!22 = distinct !DILexicalBlock(scope: !17, file: !1, line: 2, column: 14)
!23 = !DILocation(line: 3, column: 9, scope: !21)
!24 = !DILocation(line: 3, column: 7, scope: !22)
!25 = !DILocation(line: 3, column: 14, scope: !26)
!26 = !DILexicalBlockFile(scope: !21, file: !1, discriminator: 1)
!27 = !DILocation(line: 5, column: 5, scope: !28)
!28 = distinct !DILexicalBlock(scope: !21, file: !1, line: 3, column: 14)
!29 = !DILocation(line: 6, column: 3, scope: !28)
!30 = !DILocation(line: 8, column: 7, scope: !31)
!31 = distinct !DILexicalBlock(scope: !22, file: !1, line: 8, column: 7)
!32 = !DILocation(line: 8, column: 9, scope: !31)
!33 = !DILocation(line: 8, column: 7, scope: !22)
!34 = !DILocation(line: 9, column: 4, scope: !35)
!35 = distinct !DILexicalBlock(scope: !31, file: !1, line: 8, column: 14)
!36 = !DILocation(line: 11, column: 7, scope: !22)
!37 = !DILocation(line: 11, column: 9, scope: !22)
!38 = !DILocation(line: 11, column: 5, scope: !22)
!39 = !DILocation(line: 12, column: 2, scope: !22)
!40 = !DILocation(line: 13, column: 1, scope: !4)
