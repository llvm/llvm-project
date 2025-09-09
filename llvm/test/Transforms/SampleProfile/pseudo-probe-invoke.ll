; REQUIRES: x86-registered-target
; RUN: opt < %s -passes=pseudo-probe -S -o - | FileCheck %s

$__clang_call_terminate = comdat any

@x = dso_local global i32 0, align 4, !dbg !0

; Function Attrs: mustprogress noinline nounwind uwtable
define dso_local void @_Z3barv() #0 personality ptr @__gxx_personality_v0 !dbg !14 {
entry:
; CHECK: call void @llvm.pseudoprobe(i64 -1069303473483922844, i64 1
  %0 = load volatile i32, ptr @x, align 4, !dbg !17, !tbaa !19
  %tobool = icmp ne i32 %0, 0, !dbg !17
  br i1 %tobool, label %if.then, label %if.else, !dbg !23

if.then:                                          ; preds = %entry
; CHECK: call void @llvm.pseudoprobe(i64 -1069303473483922844, i64 2
; callsite probe 3
  invoke void @_Z3foov()
          to label %invoke.cont unwind label %terminate.lpad, !dbg !24

invoke.cont:                                      ; preds = %if.then
; callsite probe 4
; CHECK-NOT: call void @llvm.pseudoprobe(i64 -1069303473483922844,
  invoke void @_Z3bazv()
          to label %invoke.cont1 unwind label %terminate.lpad, !dbg !26

invoke.cont1:                                     ; preds = %invoke.cont
; CHECK-NOT: call void @llvm.pseudoprobe(i64 -1069303473483922844,
  br label %if.end, !dbg !27

if.else:                                          ; preds = %entry
; CHECK: call void @llvm.pseudoprobe(i64 -1069303473483922844, i64 5
; callsite probe 6
  invoke void @_Z3foov()
          to label %invoke.cont2 unwind label %terminate.lpad, !dbg !28

invoke.cont2:                                     ; preds = %if.else
; CHECK-NOT: call void @llvm.pseudoprobe(i64 -1069303473483922844,
  br label %if.end

if.end:                                           ; preds = %invoke.cont2, %invoke.cont1
; CHECK: call void @llvm.pseudoprobe(i64 -1069303473483922844, i64 7
; callsite probe 8
  invoke void @_Z3foov()
          to label %invoke.cont3 unwind label %terminate.lpad, !dbg !29

invoke.cont3:                                     ; preds = %if.end
; CHECK-NOT: call void @llvm.pseudoprobe(i64 -1069303473483922844,
  %1 = load volatile i32, ptr @x, align 4, !dbg !30, !tbaa !19
  %tobool4 = icmp ne i32 %1, 0, !dbg !30
  br i1 %tobool4, label %if.then5, label %if.end6, !dbg !32

if.then5:                                         ; preds = %invoke.cont3
; CHECK: call void @llvm.pseudoprobe(i64 -1069303473483922844, i64 9
  %2 = load volatile i32, ptr @x, align 4, !dbg !33, !tbaa !19
  %inc = add nsw i32 %2, 1, !dbg !33
  store volatile i32 %inc, ptr @x, align 4, !dbg !33, !tbaa !19
  br label %if.end6, !dbg !35

if.end6:                                          ; preds = %if.then5, %invoke.cont3
; CHECK: call void @llvm.pseudoprobe(i64 -1069303473483922844, i64 10
  ret void, !dbg !36

terminate.lpad:                                   ; preds = %if.end, %if.else, %invoke.cont, %if.then
; CHECK-NOT: call void @llvm.pseudoprobe(i64 -1069303473483922844,
  %3 = landingpad { ptr, i32 }
          catch ptr null, !dbg !24
  %4 = extractvalue { ptr, i32 } %3, 0, !dbg !24
  call void @__clang_call_terminate(ptr %4) #3, !dbg !24
  unreachable, !dbg !24
}

; Function Attrs: mustprogress noinline nounwind uwtable
define dso_local void @_Z3foov() #0 !dbg !37 {
entry:
  ret void, !dbg !38
}

declare i32 @__gxx_personality_v0(...)

; Function Attrs: noinline noreturn nounwind uwtable
define linkonce_odr hidden void @__clang_call_terminate(ptr noundef %0) #1 comdat {
  %2 = call ptr @__cxa_begin_catch(ptr %0) #4
  call void @_ZSt9terminatev() #3
  unreachable
}

declare ptr @__cxa_begin_catch(ptr)

declare void @_ZSt9terminatev()

; Function Attrs: mustprogress noinline nounwind uwtable
define dso_local void @_Z3bazv() #0 !dbg !39 {
entry:
  ret void, !dbg !40
}

; CHECK: ![[#]] = !{i64 -3270123626113159616, i64 4294967295, !"_Z3bazv"}

attributes #0 = { mustprogress noinline nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { noinline noreturn nounwind uwtable "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { mustprogress noinline norecurse nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { noreturn nounwind }
attributes #4 = { nounwind }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9, !10, !11, !12}
!llvm.ident = !{!13}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "x", scope: !2, file: !3, line: 1, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 19.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "test.cpp", directory: "/home", checksumkind: CSK_MD5, checksum: "a4c7b0392f3fd9c8ebb85065159dbb02")
!4 = !{!0}
!5 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !6)
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{i32 7, !"Dwarf Version", i32 5}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{i32 8, !"PIC Level", i32 2}
!11 = !{i32 7, !"PIE Level", i32 2}
!12 = !{i32 7, !"uwtable", i32 2}
!13 = !{!"clang version 19.0.0"}
!14 = distinct !DISubprogram(name: "bar", linkageName: "_Z3barv", scope: !3, file: !3, line: 4, type: !15, scopeLine: 4, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2)
!15 = !DISubroutineType(types: !16)
!16 = !{null}
!17 = !DILocation(line: 5, column: 6, scope: !18)
!18 = distinct !DILexicalBlock(scope: !14, file: !3, line: 5, column: 6)
!19 = !{!20, !20, i64 0}
!20 = !{!"int", !21, i64 0}
!21 = !{!"omnipotent char", !22, i64 0}
!22 = !{!"Simple C++ TBAA"}
!23 = !DILocation(line: 5, column: 6, scope: !14)
!24 = !DILocation(line: 6, column: 5, scope: !25)
!25 = distinct !DILexicalBlock(scope: !18, file: !3, line: 5, column: 9)
!26 = !DILocation(line: 7, column: 5, scope: !25)
!27 = !DILocation(line: 8, column: 3, scope: !25)
!28 = !DILocation(line: 9, column: 5, scope: !18)
!29 = !DILocation(line: 11, column: 3, scope: !14)
!30 = !DILocation(line: 12, column: 6, scope: !31)
!31 = distinct !DILexicalBlock(scope: !14, file: !3, line: 12, column: 6)
!32 = !DILocation(line: 12, column: 6, scope: !14)
!33 = !DILocation(line: 13, column: 5, scope: !34)
!34 = distinct !DILexicalBlock(scope: !31, file: !3, line: 12, column: 9)
!35 = !DILocation(line: 14, column: 5, scope: !34)
!36 = !DILocation(line: 17, column: 1, scope: !14)
!37 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !3, file: !3, line: 19, type: !15, scopeLine: 19, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2)
!38 = !DILocation(line: 19, column: 13, scope: !37)
!39 = distinct !DISubprogram(name: "baz", linkageName: "_Z3bazv", scope: !3, file: !3, line: 18, type: !15, scopeLine: 18, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2)
!40 = !DILocation(line: 18, column: 13, scope: !39)
!41 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 22, type: !42, scopeLine: 22, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2)
!42 = !DISubroutineType(types: !43)
!43 = !{!6}
!44 = !DILocation(line: 23, column: 3, scope: !41)
!45 = !DILocation(line: 24, column: 1, scope: !41)
