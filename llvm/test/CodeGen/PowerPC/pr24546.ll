; RUN: llc -verify-machineinstrs -mcpu=pwr8 -mtriple=powerpc64le-unknown-linux-gnu < %s

; Verify that we no longer crash in VSX swap removal when debug values
; are in the code stream.

@php_intpow10.powers = external unnamed_addr constant [23 x double], align 8, !dbg !24

; Function Attrs: nounwind
define double @_php_math_round(double %value, i32 signext %places, i32 signext %mode) #0 !dbg !6 {
entry:
  br i1 undef, label %if.then, label %if.else, !dbg !32

if.then:                                          ; preds = %entry
  %conv = sitofp i32 undef to double, !dbg !34
  br i1 undef, label %if.then.i, label %if.end.i, !dbg !36

if.then.i:                                        ; preds = %if.then
  %call.i = tail call double @pow(double 1.000000e+01, double undef) #3, !dbg !39
  br label %php_intpow10.exit, !dbg !41

if.end.i:                                         ; preds = %if.then
  %0 = load double, double* undef, align 8, !dbg !42, !tbaa !43
  br label %php_intpow10.exit, !dbg !47

php_intpow10.exit:                                ; preds = %if.end.i, %if.then.i
  %retval.0.i = phi double [ %call.i, %if.then.i ], [ %0, %if.end.i ], !dbg !48
  tail call void @llvm.dbg.value(metadata double %retval.0.i, i64 0, metadata !15, metadata !49), !dbg !50
  %div = fdiv double %conv, %retval.0.i, !dbg !51
  br label %if.end.15, !dbg !52

if.else:                                          ; preds = %entry
  %mul = fmul double %value, undef, !dbg !53
  br label %if.end.15

if.end.15:                                        ; preds = %if.else, %php_intpow10.exit
  %tmp_value.1 = phi double [ %div, %php_intpow10.exit ], [ %mul, %if.else ]
  ret double %tmp_value.1, !dbg !57
}

declare signext i32 @php_intlog10abs(...) #1

declare signext i32 @php_round_helper(...) #1

; Function Attrs: nounwind
declare double @pow(double, double) #0

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #2

attributes #0 = { nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="ppc64le" "target-features"="+altivec,+bpermd,+crypto,+direct-move,+extdiv,+power8-vector,+vsx,-qpx" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="ppc64le" "target-features"="+altivec,+bpermd,+crypto,+direct-move,+extdiv,+power8-vector,+vsx,-qpx" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!29, !30}
!llvm.ident = !{!31}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.8.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, globals: !23)
!1 = !DIFile(filename: "testcase.i", directory: "/tmp/glibc.build")
!2 = !{}
!3 = !{!4}
!4 = !DIBasicType(name: "double", size: 64, align: 64, encoding: DW_ATE_float)
!6 = distinct !DISubprogram(name: "_php_math_round", scope: !1, file: !1, line: 15, type: !7, isLocal: false, isDefinition: true, scopeLine: 16, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !10)
!7 = !DISubroutineType(types: !8)
!8 = !{!4, !4, !9, !9}
!9 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !{!11, !12, !13, !14, !15, !16, !17}
!11 = !DILocalVariable(name: "value", arg: 1, scope: !6, file: !1, line: 15, type: !4)
!12 = !DILocalVariable(name: "places", arg: 2, scope: !6, file: !1, line: 15, type: !9)
!13 = !DILocalVariable(name: "mode", arg: 3, scope: !6, file: !1, line: 15, type: !9)
!14 = !DILocalVariable(name: "f1", scope: !6, file: !1, line: 17, type: !4)
!15 = !DILocalVariable(name: "f2", scope: !6, file: !1, line: 17, type: !4)
!16 = !DILocalVariable(name: "tmp_value", scope: !6, file: !1, line: 18, type: !4)
!17 = !DILocalVariable(name: "precision_places", scope: !6, file: !1, line: 19, type: !9)
!18 = distinct !DISubprogram(name: "php_intpow10", scope: !1, file: !1, line: 1, type: !19, isLocal: true, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !21)
!19 = !DISubroutineType(types: !20)
!20 = !{!4, !9}
!21 = !{!22}
!22 = !DILocalVariable(name: "power", arg: 1, scope: !18, file: !1, line: 1, type: !9)
!23 = !{!24}
!24 = !DIGlobalVariableExpression(var: !DIGlobalVariable(name: "powers", scope: !18, file: !1, line: 3, type: !25, isLocal: true, isDefinition: true), expr: !DIExpression())
!25 = !DICompositeType(tag: DW_TAG_array_type, baseType: !26, size: 1472, align: 64, elements: !27)
!26 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !4)
!27 = !{!28}
!28 = !DISubrange(count: 23)
!29 = !{i32 2, !"Dwarf Version", i32 4}
!30 = !{i32 2, !"Debug Info Version", i32 3}
!31 = !{!"clang version 3.8.0"}
!32 = !DILocation(line: 21, column: 32, scope: !33)
!33 = distinct !DILexicalBlock(scope: !6, file: !1, line: 21, column: 6)
!34 = !DILocation(line: 22, column: 15, scope: !35)
!35 = distinct !DILexicalBlock(scope: !33, file: !1, line: 21, column: 67)
!36 = !DILocation(line: 8, column: 16, scope: !37, inlinedAt: !38)
!37 = distinct !DILexicalBlock(scope: !18, file: !1, line: 8, column: 6)
!38 = distinct !DILocation(line: 23, column: 8, scope: !35)
!39 = !DILocation(line: 9, column: 10, scope: !40, inlinedAt: !38)
!40 = distinct !DILexicalBlock(scope: !37, file: !1, line: 8, column: 31)
!41 = !DILocation(line: 9, column: 3, scope: !40, inlinedAt: !38)
!42 = !DILocation(line: 11, column: 9, scope: !18, inlinedAt: !38)
!43 = !{!44, !44, i64 0}
!44 = !{!"double", !45, i64 0}
!45 = !{!"omnipotent char", !46, i64 0}
!46 = !{!"Simple C/C++ TBAA"}
!47 = !DILocation(line: 11, column: 2, scope: !18, inlinedAt: !38)
!48 = !DILocation(line: 23, column: 8, scope: !35)
!49 = !DIExpression()
!50 = !DILocation(line: 17, column: 13, scope: !6)
!51 = !DILocation(line: 24, column: 25, scope: !35)
!52 = !DILocation(line: 25, column: 2, scope: !35)
!53 = !DILocation(line: 27, column: 22, scope: !54)
!54 = distinct !DILexicalBlock(scope: !55, file: !1, line: 26, column: 20)
!55 = distinct !DILexicalBlock(scope: !56, file: !1, line: 26, column: 7)
!56 = distinct !DILexicalBlock(scope: !33, file: !1, line: 25, column: 9)
!57 = !DILocation(line: 32, column: 2, scope: !6)
