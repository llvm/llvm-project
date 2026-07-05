; RUN: llc -mtriple=x86_64-unknown-linux-gnu -start-after=codegenprepare -stop-before=finalize-isel -o - %s -experimental-debug-variable-locations=false | FileCheck %s
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -start-after=codegenprepare -stop-before=finalize-isel -o - %s -experimental-debug-variable-locations=true | FileCheck %s --check-prefixes=INSTRREF

; Input to this test looked like this and was compiled using: clang -g -O1 -mllvm -stop-after=codegenprepare -S
;
;    int fn1(long t1) {
;      return t;
;    }
;

; Catch metadata references for involved variables.
;
; CHECK-DAG: ![[T1:.*]] = !DILocalVariable(name: "t1"
; INSTRREF-DAG: ![[T1:.*]] = !DILocalVariable(name: "t1"


define dso_local i32 @fn1(i64 %t1) local_unnamed_addr #0 !dbg !7 {
; We expect that the same width COPY reuses the debug location,
; but the width narrowing COPY does not.
; 
; CHECK-LABEL: name:            fn1
; CHECK: DBG_VALUE $rdi, $noreg, ![[T1]], !DIExpression(),
; CHECK-NEXT: %0:gr64 = COPY $rdi
; CHECK-NEXT: DBG_VALUE %0, $noreg, ![[T1]], !DIExpression(),
; CHECK-NEXT: %1:gr32 = COPY %0.sub_32bit
; CHECK-NEXT: COPY
; CHECK-NEXT: RET
;
;; For instr-ref, no copies should be considered. Because argumenst are
;; Special, we don't label them in the same way, and currently emit a
;; DBG_VALUE for the physreg.
; INSTRREF-LABEL: name:            fn1
; INSTRREF: DBG_VALUE $rdi, $noreg, ![[T1]], !DIExpression(),
; INSTRREF-NEXT: %0:gr64 = COPY $rdi
; INSTRREF-NEXT: %1:gr32 = COPY %0.sub_32bit
; INSTRREF-NEXT: COPY
; INSTRREF-NEXT: RET

entry:
  call void @llvm.dbg.value(metadata i64 %t1, metadata !13, metadata !DIExpression()), !dbg !14
  %0 = trunc i64 %t1 to i32, !dbg !15
  ret i32 %0, !dbg !16
}

declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { norecurse nounwind readnone uwtable }
attributes #1 = { nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 11.0.0 (git@github.com:tbosch/llvm-project.git 0b11aed869bf09ba60
d7ed17334cf0b76e6a5922)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cc", directory: "")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 11.0.0 (git@github.com:tbosch/llvm-project.git 0b11aed869bf09ba60d7ed17334cf0b76e6a5922)"}
!7 = distinct !DISubprogram(name: "fn1", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !12)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !11}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DIBasicType(name: "long int", size: 64, encoding: DW_ATE_signed)
!12 = !{!13}
!13 = !DILocalVariable(name: "t1", arg: 1, scope: !7, file: !1, line: 1, type: !11)
!14 = !DILocation(line: 0, scope: !7)
!15 = !DILocation(line: 2, column: 10, scope: !7)
!16 = !DILocation(line: 2, column: 3, scope: !7)
