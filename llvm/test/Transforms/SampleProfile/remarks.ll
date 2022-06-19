; RUN: opt < %s -sample-profile -sample-profile-file=%S/Inputs/remarks.prof -S -pass-remarks=sample-profile -pass-remarks-output=%t.opt.yaml 2>&1 | FileCheck %s
; RUN: FileCheck %s -check-prefix=YAML < %t.opt.yaml
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/remarks.prof -S -pass-remarks=sample-profile -pass-remarks-output=%t.opt.yaml 2>&1 | FileCheck %s
; RUN: FileCheck %s -check-prefix=YAML < %t.opt.yaml

; Original test case.
;
;     1    #include <stdlib.h>
;     2
;     3    long long foo() {
;     4      long long int sum = 0;
;     5      for (int i = 0; i < 500000000; i++)
;     6        if (i < 1000)
;     7          sum -= i;
;     8        else
;     9          sum += -i * rand();
;    10      return sum;
;    11    }
;    12
;    13    int main() { return foo() > 0; }

; We are expecting foo() to be inlined in main() (almost all the cycles are
; spent inside foo).
; CHECK: remark: remarks.cc:13:21: '_Z3foov' inlined into 'main' to match profiling context with (cost=130, threshold=2147483647) at callsite main:0:21;
; CHECK: remark: remarks.cc:9:19: 'rand' inlined into 'main' to match profiling context with (cost=always): always inline attribute at callsite _Z3foov:6:19 @ main:0:21;

; The back edge for the loop is the hottest edge in the loop subgraph.
; CHECK: remark: remarks.cc:6:9: most popular destination for conditional branches at remarks.cc:5:3

; The predicate almost always chooses the 'else' branch.
; CHECK: remark: remarks.cc:9:15: most popular destination for conditional branches at remarks.cc:6:9

; Checking to see if YAML file is generated and contains remarks
;YAML:       --- !Passed
;YAML-NEXT:  Pass:            sample-profile-inline
;YAML-NEXT:  Name:            Inlined
;YAML-NEXT:  DebugLoc:        { File: remarks.cc, Line: 13, Column: 21 }
;YAML-NEXT:  Function:        main
;YAML-NEXT:  Args:
;YAML-NEXT:    - String:          ''''
;YAML-NEXT:    - Callee:          _Z3foov
;YAML-NEXT:      DebugLoc:        { File: remarks.cc, Line: 3, Column: 0 }
;YAML-NEXT:    - String:          ''' inlined into '
;YAML-NEXT:    - Caller:          main
;YAML-NEXT:        DebugLoc:        { File: remarks.cc, Line: 13, Column: 0 }
;YAML-NEXT:    - String:          ''''
;YAML-NEXT:    - String:          ' to match profiling context'
;YAML-NEXT:    - String:          ' with '
;YAML-NEXT:    - String:          '(cost='
;YAML-NEXT:    - Cost:            '130'
;YAML-NEXT:    - String:          ', threshold='
;YAML-NEXT:    - Threshold:       '2147483647'
;YAML-NEXT:    - String:          ')'
;YAML-NEXT:    - String:          ' at callsite '
;YAML-NEXT:    - String:          main
;YAML-NEXT:    - String:          ':'
;YAML-NEXT:    - Line:            '0'
;YAML-NEXT:    - String:          ':'
;YAML-NEXT:    - Column:          '21'
;YAML-NEXT:    - String:          ';'
;YAML-NEXT:  ...
;YAML:       --- !Passed
;YAML-NEXT:  Pass:            sample-profile-inline
;YAML-NEXT:  Name:            AlwaysInline
;YAML-NEXT:  DebugLoc:        { File: remarks.cc, Line: 9, Column: 19 }
;YAML-NEXT:  Function:        main
;YAML-NEXT:  Args:
;YAML-NEXT:    - String:          ''''
;YAML-NEXT:    - Callee:          rand
;YAML-NEXT:      DebugLoc:        { File: remarks.cc, Line: 90, Column: 0 }
;YAML-NEXT:    - String:          ''' inlined into '''
;YAML-NEXT:    - Caller:          main
;YAML-NEXT:      DebugLoc:        { File: remarks.cc, Line: 13, Column: 0 }
;YAML-NEXT:    - String:          ''''
;YAML-NEXT:    - String:          ' to match profiling context'
;YAML-NEXT:    - String:          ' with '
;YAML-NEXT:    - String:          '(cost=always)'
;YAML-NEXT:    - String:          ': '
;YAML-NEXT:    - Reason:          always inline attribute
;YAML-NEXT:    - String:          ' at callsite '
;YAML-NEXT:    - String:          _Z3foov
;YAML-NEXT:    - String:          ':'
;YAML-NEXT:    - Line:            '6'
;YAML-NEXT:    - String:          ':'
;YAML-NEXT:    - Column:          '19'
;YAML-NEXT:    - String:          ' @ '
;YAML-NEXT:    - String:          main
;YAML-NEXT:    - String:          ':'
;YAML-NEXT:    - Line:            '0'
;YAML-NEXT:    - String:          ':'
;YAML-NEXT:    - Column:          '21'
;YAML-NEXT:    - String:          ';'
;YAML:  --- !Analysis
;YAML-NEXT:  Pass:            sample-profile
;YAML-NEXT:  Name:            AppliedSamples
;YAML-NEXT:  DebugLoc:        { File: remarks.cc, Line: 5, Column: 8 }
;YAML-NEXT:  Function:        main
;YAML-NEXT:  Args:
;YAML-NEXT:    - String:          'Applied '
;YAML-NEXT:    - NumSamples:      '18305'
;YAML-NEXT:    - String:          ' samples from profile (offset: '
;YAML-NEXT:    - LineOffset:      '2'
;YAML-NEXT:    - String:          ')'
;YAML-NEXT:  ...
;YAML:  --- !Passed
;YAML-NEXT:  Pass:            sample-profile
;YAML-NEXT:  Name:            PopularDest
;YAML-NEXT:  DebugLoc:        { File: remarks.cc, Line: 6, Column: 9 }
;YAML-NEXT:  Function:        main
;YAML-NEXT:  Args:
;YAML-NEXT:    - String:          'most popular destination for conditional branches at '
;YAML-NEXT:    - CondBranchesLoc: 'remarks.cc:5:3'
;YAML-NEXT:      DebugLoc:        { File: remarks.cc, Line: 5, Column: 3 }
;YAML-NEXT:  ...

; Function Attrs: nounwind uwtable
define i64 @_Z3foov() #0 !dbg !4 {
entry:
  %sum = alloca i64, align 8
  %i = alloca i32, align 4
  %0 = bitcast i64* %sum to i8*, !dbg !19
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %0) #4, !dbg !19
  call void @llvm.dbg.declare(metadata i64* %sum, metadata !9, metadata !20), !dbg !21
  store i64 0, i64* %sum, align 8, !dbg !21, !tbaa !22
  %1 = bitcast i32* %i to i8*, !dbg !26
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %1) #4, !dbg !26
  call void @llvm.dbg.declare(metadata i32* %i, metadata !10, metadata !20), !dbg !27
  store i32 0, i32* %i, align 4, !dbg !27, !tbaa !28
  br label %for.cond, !dbg !26

for.cond:                                         ; preds = %for.inc, %entry
  %2 = load i32, i32* %i, align 4, !dbg !30, !tbaa !28
  %cmp = icmp slt i32 %2, 500000000, !dbg !34
  br i1 %cmp, label %for.body, label %for.cond.cleanup, !dbg !35

for.cond.cleanup:                                 ; preds = %for.cond
  %3 = bitcast i32* %i to i8*, !dbg !36
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %3) #4, !dbg !36
  br label %for.end

for.body:                                         ; preds = %for.cond
  %4 = load i32, i32* %i, align 4, !dbg !38, !tbaa !28
  %cmp1 = icmp slt i32 %4, 1000, !dbg !40
  br i1 %cmp1, label %if.then, label %if.else, !dbg !41

if.then:                                          ; preds = %for.body
  %5 = load i32, i32* %i, align 4, !dbg !42, !tbaa !28
  %conv = sext i32 %5 to i64, !dbg !42
  %6 = load i64, i64* %sum, align 8, !dbg !43, !tbaa !22
  %sub = sub nsw i64 %6, %conv, !dbg !43
  store i64 %sub, i64* %sum, align 8, !dbg !43, !tbaa !22
  br label %if.end, !dbg !44

if.else:                                          ; preds = %for.body
  %7 = load i32, i32* %i, align 4, !dbg !45, !tbaa !28
  %sub2 = sub nsw i32 0, %7, !dbg !46
  %call = call i32 @rand() #4, !dbg !47
  %mul = mul nsw i32 %sub2, %call, !dbg !48
  %conv3 = sext i32 %mul to i64, !dbg !46
  %8 = load i64, i64* %sum, align 8, !dbg !49, !tbaa !22
  %add = add nsw i64 %8, %conv3, !dbg !49
  store i64 %add, i64* %sum, align 8, !dbg !49, !tbaa !22
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  br label %for.inc, !dbg !50

for.inc:                                          ; preds = %if.end
  %9 = load i32, i32* %i, align 4, !dbg !51, !tbaa !28
  %inc = add nsw i32 %9, 1, !dbg !51
  store i32 %inc, i32* %i, align 4, !dbg !51, !tbaa !28
  br label %for.cond, !dbg !52

for.end:                                          ; preds = %for.cond.cleanup
  %10 = load i64, i64* %sum, align 8, !dbg !53, !tbaa !22
  %11 = bitcast i64* %sum to i8*, !dbg !54
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %11) #4, !dbg !54
  ret i64 %10, !dbg !55
}

; Function Attrs: nounwind argmemonly
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

; Function Attrs: nounwind
define i32 @rand() #3 !dbg !59 {
  ret i32 1
}

; Function Attrs: nounwind argmemonly
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

; Function Attrs: nounwind uwtable
define i32 @main() #0 !dbg !13 {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  %call = call i64 @_Z3foov(), !dbg !56
  %cmp = icmp sgt i64 %call, 0, !dbg !57
  %conv = zext i1 %cmp to i32, !dbg !56
  ret i32 %conv, !dbg !58
}

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" "use-sample-profile" }
attributes #1 = { nounwind argmemonly }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind alwaysinline "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!16, !17}
!llvm.ident = !{!18}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.8.0 (trunk 251041) (llvm/trunk 251053)", isOptimized: true, runtimeVersion: 0, emissionKind: NoDebug, enums: !2)
!1 = !DIFile(filename: "remarks.cc", directory: ".")
!2 = !{}
!4 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !1, file: !1, line: 3, type: !5, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !8)
!5 = !DISubroutineType(types: !6)
!6 = !{!7}
!7 = !DIBasicType(name: "long long int", size: 64, align: 64, encoding: DW_ATE_signed)
!8 = !{!9, !10}
!9 = !DILocalVariable(name: "sum", scope: !4, file: !1, line: 4, type: !7)
!10 = !DILocalVariable(name: "i", scope: !11, file: !1, line: 5, type: !12)
!11 = distinct !DILexicalBlock(scope: !4, file: !1, line: 5, column: 3)
!12 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!13 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 13, type: !14, isLocal: false, isDefinition: true, scopeLine: 13, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !2)
!14 = !DISubroutineType(types: !15)
!15 = !{!12}
!16 = !{i32 2, !"Dwarf Version", i32 4}
!17 = !{i32 2, !"Debug Info Version", i32 3}
!18 = !{!"clang version 3.8.0 (trunk 251041) (llvm/trunk 251053)"}
!19 = !DILocation(line: 4, column: 3, scope: !4)
!20 = !DIExpression()
!21 = !DILocation(line: 4, column: 17, scope: !4)
!22 = !{!23, !23, i64 0}
!23 = !{!"long long", !24, i64 0}
!24 = !{!"omnipotent char", !25, i64 0}
!25 = !{!"Simple C/C++ TBAA"}
!26 = !DILocation(line: 5, column: 8, scope: !11)
!27 = !DILocation(line: 5, column: 12, scope: !11)
!28 = !{!29, !29, i64 0}
!29 = !{!"int", !24, i64 0}
!30 = !DILocation(line: 5, column: 19, scope: !31)
!31 = !DILexicalBlockFile(scope: !32, file: !1, discriminator: 3)
!32 = !DILexicalBlockFile(scope: !33, file: !1, discriminator: 1)
!33 = distinct !DILexicalBlock(scope: !11, file: !1, line: 5, column: 3)
!34 = !DILocation(line: 5, column: 21, scope: !33)
!35 = !DILocation(line: 5, column: 3, scope: !11)
!36 = !DILocation(line: 5, column: 3, scope: !37)
!37 = !DILexicalBlockFile(scope: !33, file: !1, discriminator: 2)
!38 = !DILocation(line: 6, column: 9, scope: !39)
!39 = distinct !DILexicalBlock(scope: !33, file: !1, line: 6, column: 9)
!40 = !DILocation(line: 6, column: 11, scope: !39)
!41 = !DILocation(line: 6, column: 9, scope: !33)
!42 = !DILocation(line: 7, column: 14, scope: !39)
!43 = !DILocation(line: 7, column: 11, scope: !39)
!44 = !DILocation(line: 7, column: 7, scope: !39)
!45 = !DILocation(line: 9, column: 15, scope: !39)
!46 = !DILocation(line: 9, column: 14, scope: !39)
!47 = !DILocation(line: 9, column: 19, scope: !39)
!48 = !DILocation(line: 9, column: 17, scope: !39)
!49 = !DILocation(line: 9, column: 11, scope: !39)
!50 = !DILocation(line: 6, column: 13, scope: !39)
!51 = !DILocation(line: 5, column: 35, scope: !33)
!52 = !DILocation(line: 5, column: 3, scope: !33)
!53 = !DILocation(line: 10, column: 10, scope: !4)
!54 = !DILocation(line: 11, column: 1, scope: !4)
!55 = !DILocation(line: 10, column: 3, scope: !4)
!56 = !DILocation(line: 13, column: 21, scope: !13)
!57 = !DILocation(line: 13, column: 27, scope: !13)
!58 = !DILocation(line: 13, column: 14, scope: !13)
!59 = distinct !DISubprogram(name: "rand", linkageName: "rand", scope: !1, file: !1, line: 90, type: !5, isLocal: false, isDefinition: true, scopeLine: 90, flags: DIFlagPrototyped, isOptimized: true, unit: !0)
