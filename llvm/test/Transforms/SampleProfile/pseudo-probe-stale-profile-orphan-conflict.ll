; Test direct basename matching for orphan functions where multiple callee anchors may be
; matched to one same profile during stale profile matching. In this test case, both of the
; `mid` functions will be matched to the same `_Z3midi` function in the profile during stale
; profile matching. This ends up causing an assertation error because only one profile
; function is supposed to be matched to an IR function.
;
; IR Function:
;   foo: top ; bar ; top(2)
;         |_mid       |_ mid(2)
;                          |_ sub
; 
; Profile Function:
;   foo: bar ; top ; mid
;                     |_ sub
; 
; Stale profile match order:
;   foo:  top<ir>   ; bar<ir>   ; top(2)<ir>
;            |           |           |
;         top<prof> ; bar<prof> ; top<prof>
;
;   top(2)<ir>:  mid(2)<ir>
;                  |
;                mid<prof>
;
;   top<ir>:  mid<ir>
;               |
;             mid<prof>    => (Assertation error)


; REQUIRES: x86_64-linux
; REQUIRES: asserts
; RUN: llvm-profdata merge --sample --extbinary %S/Inputs/pseudo-probe-stale-profile-orphan-conflict.prof -o %t.prof
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%t.prof --salvage-stale-profile --salvage-unused-profile -S --debug-only=sample-profile,sample-profile-matcher,sample-profile-impl 2>&1 | FileCheck %s

; CHECK: Function _Z3midl is not in profile or profile symbol list.
; CHECK: Function _Z3midll is not in profile or profile symbol list.
; CHECK: Function _Z3topl is not in profile or profile symbol list.
; CHECK: Function _Z3barl is not in profile or profile symbol list.
; CHECK: Function _Z3topll is not in profile or profile symbol list.
; CHECK: Function _Z3fool is not in profile or profile symbol list.
; CHECK: Direct basename match: _Z3barl (IR) -> _Z3bari (Profile) [basename: bar]
; CHECK: Direct basename match: _Z3fool (IR) -> _Z3fooi (Profile) [basename: foo]
; CHECK: Direct basename matching found 2 matches
; CHECK: Run stale profile matching for _Z3fool
; CHECK: The functions _Z3topl(IR) and _Z3topi(Profile) share the same base name: top.
; CHECK: Function:_Z3topl matches profile:_Z3topi
; CHECK: The functions _Z3barl(IR) and _Z3bari(Profile) share the same base name: bar.
; CHECK: Function:_Z3barl matches profile:_Z3bari
; CHECK: The functions _Z3topll(IR) and _Z3topi(Profile) share the same base name: top.
; CHECK: Function:_Z3topll matches profile:_Z3topi
; CHECK: Location is matched from 2 to 2
; CHECK: Callsite with callee:_Z3barl is matched from 3 to 57
; CHECK: Callsite with callee:_Z3topll is matched from 4 to 72
; CHECK: Run stale profile matching for _Z3topll
; CHECK: The functions _Z3midll(IR) and _Z3midi(Profile) share the same base name: mid.
; CHECK: Function:_Z3midll matches profile:_Z3midi
; CHECK: Callsite with callee:_Z3midll is matched from 1 to 2
; CHECK: Run stale profile matching for _Z3barl
; CHECK: Run stale profile matching for _Z3topl
; CHECK: The functions _Z3midl(IR) and _Z3midi(Profile) share the same base name: mid.
; CHECK: Function:_Z3midl matches profile:_Z3midi
; CHECK: Callsite with callee:_Z3midl is matched from 1 to 2
; CHECK: Run stale profile matching for _Z3midll
; CHECK: Callsite with callee:_Z3subi is matched from 1 to 11
; CHECK: Run stale profile matching for _Z3subi
; CHECK: Run stale profile matching for _Z3midl
; CHECK: Function processing order:
; CHECK: _Z3topll
; CHECK: _Z3midl
; CHECK: _Z3fool
; CHECK: _Z3topl
; CHECK: _Z3midll
; CHECK: _Z3subi
; CHECK: _Z3barl

target triple = "x86_64-redhat-linux-gnu"

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local noundef ptr @_Z3midl(i64 noundef %l) #0 !dbg !14 {
entry:
  %l.addr = alloca i64, align 8
  store i64 %l, ptr %l.addr, align 8
    #dbg_declare(ptr %l.addr, !21, !DIExpression(), !22)
  call void @llvm.pseudoprobe(i64 -4458821264266946817, i64 1, i32 0, i64 -1), !dbg !23
  ret ptr null, !dbg !23
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local noundef ptr @_Z3subi(i32 noundef %i) #0 !dbg !24 {
entry:
  %i.addr = alloca i32, align 4
  store i32 %i, ptr %i.addr, align 4
    #dbg_declare(ptr %i.addr, !28, !DIExpression(), !29)
  call void @llvm.pseudoprobe(i64 8307782004441981189, i64 1, i32 0, i64 -1), !dbg !30
  ret ptr null, !dbg !30
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local noundef ptr @_Z3midll(i64 noundef %l, i64 noundef %m) #0 !dbg !31 {
entry:
  %l.addr = alloca i64, align 8
  %m.addr = alloca i64, align 8
  store i64 %l, ptr %l.addr, align 8
    #dbg_declare(ptr %l.addr, !34, !DIExpression(), !35)
  store i64 %m, ptr %m.addr, align 8
    #dbg_declare(ptr %m.addr, !36, !DIExpression(), !37)
  call void @llvm.pseudoprobe(i64 -835688601043669768, i64 1, i32 0, i64 -1), !dbg !38
  %call = call noundef ptr @_Z3subi(i32 noundef 0), !dbg !39
  ret ptr %call, !dbg !41
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local noundef ptr @_Z3topl(i64 noundef %l) #0 !dbg !42 {
entry:
  %l.addr = alloca i64, align 8
  store i64 %l, ptr %l.addr, align 8
    #dbg_declare(ptr %l.addr, !43, !DIExpression(), !44)
  call void @llvm.pseudoprobe(i64 7421866232655760046, i64 1, i32 0, i64 -1), !dbg !45
  %0 = load i64, ptr %l.addr, align 8, !dbg !45
  %call = call noundef ptr @_Z3midl(i64 noundef %0), !dbg !46
  ret ptr %call, !dbg !48
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local void @_Z3barl(i64 noundef %l) #0 !dbg !49 {
entry:
  %l.addr = alloca i64, align 8
  store i64 %l, ptr %l.addr, align 8
    #dbg_declare(ptr %l.addr, !52, !DIExpression(), !53)
  call void @llvm.pseudoprobe(i64 -9164787269840974918, i64 1, i32 0, i64 -1), !dbg !54
  ret void, !dbg !54
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local noundef ptr @_Z3topll(i64 noundef %l, i64 noundef %m) #0 !dbg !55 {
entry:
  %l.addr = alloca i64, align 8
  %m.addr = alloca i64, align 8
  store i64 %l, ptr %l.addr, align 8
    #dbg_declare(ptr %l.addr, !56, !DIExpression(), !57)
  store i64 %m, ptr %m.addr, align 8
    #dbg_declare(ptr %m.addr, !58, !DIExpression(), !59)
  call void @llvm.pseudoprobe(i64 997868883951813144, i64 1, i32 0, i64 -1), !dbg !60
  %0 = load i64, ptr %l.addr, align 8, !dbg !60
  %1 = load i64, ptr %m.addr, align 8, !dbg !61
  %call = call noundef ptr @_Z3midll(i64 noundef %0, i64 noundef %1), !dbg !62
  ret ptr %call, !dbg !64
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local noundef ptr @_Z3fool(i64 noundef %l) #0 !dbg !65 {
entry:
  %retval = alloca ptr, align 8
  %l.addr = alloca i64, align 8
  store i64 %l, ptr %l.addr, align 8
    #dbg_declare(ptr %l.addr, !66, !DIExpression(), !67)
  call void @llvm.pseudoprobe(i64 5326982120444056491, i64 1, i32 0, i64 -1), !dbg !68
  %0 = load i64, ptr %l.addr, align 8, !dbg !68
  %tobool = icmp ne i64 %0, 0, !dbg !68
  br i1 %tobool, label %if.then, label %if.end, !dbg !68

if.then:                                          ; preds = %entry
  call void @llvm.pseudoprobe(i64 5326982120444056491, i64 2, i32 0, i64 -1), !dbg !70
  %1 = load i64, ptr %l.addr, align 8, !dbg !70
  %call = call noundef ptr @_Z3topl(i64 noundef %1), !dbg !71
  store ptr %call, ptr %retval, align 8, !dbg !73
  br label %return, !dbg !73

if.end:                                           ; preds = %entry
  call void @llvm.pseudoprobe(i64 5326982120444056491, i64 4, i32 0, i64 -1), !dbg !74
  %2 = load i64, ptr %l.addr, align 8, !dbg !74
  call void @_Z3barl(i64 noundef %2), !dbg !75
  %3 = load i64, ptr %l.addr, align 8, !dbg !77
  %4 = load i64, ptr %l.addr, align 8, !dbg !78
  %call1 = call noundef ptr @_Z3topll(i64 noundef %3, i64 noundef %4), !dbg !79
  store ptr %call1, ptr %retval, align 8, !dbg !81
  br label %return, !dbg !81

return:                                           ; preds = %if.end, %if.then
  call void @llvm.pseudoprobe(i64 5326982120444056491, i64 7, i32 0, i64 -1), !dbg !82
  %5 = load ptr, ptr %retval, align 8, !dbg !82
  ret ptr %5, !dbg !82
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.pseudoprobe(i64, i64, i32, i64) #1

attributes #0 = { mustprogress noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "use-sample-profile" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5}
!llvm.ident = !{!6}
!llvm.pseudo_probe_desc = !{!7, !8, !9, !10, !11, !12, !13}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cc", directory: "", checksumkind: CSK_MD5, checksum: "44fecbd11c1385709b8c0c240594ca47")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 7, !"uwtable", i32 2}
!5 = !{i32 7, !"frame-pointer", i32 2}
!6 = !{!"clang"}
!7 = !{i64 -4458821264266946817, i64 4294967295, !"_Z3midl"}
!8 = !{i64 8307782004441981189, i64 4294967295, !"_Z3subi"}
!9 = !{i64 -835688601043669768, i64 281479271677951, !"_Z3midll"}
!10 = !{i64 7421866232655760046, i64 281479271677951, !"_Z3topl"}
!11 = !{i64 -9164787269840974918, i64 4294967295, !"_Z3barl"}
!12 = !{i64 997868883951813144, i64 281479271677951, !"_Z3topll"}
!13 = !{i64 5326982120444056491, i64 844493665377046, !"_Z3fool"}
!14 = distinct !DISubprogram(name: "mid", linkageName: "_Z3midl", scope: !15, file: !15, line: 1, type: !16, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !20)
!15 = !DIFile(filename: "test.cc", directory: "", checksumkind: CSK_MD5, checksum: "44fecbd11c1385709b8c0c240594ca47")
!16 = !DISubroutineType(types: !17)
!17 = !{!18, !19}
!18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!19 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!20 = !{}
!21 = !DILocalVariable(name: "l", arg: 1, scope: !14, file: !15, line: 1, type: !19)
!22 = !DILocation(line: 1, column: 16, scope: !14)
!23 = !DILocation(line: 2, column: 3, scope: !14)
!24 = distinct !DISubprogram(name: "sub", linkageName: "_Z3subi", scope: !15, file: !15, line: 5, type: !25, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !20)
!25 = !DISubroutineType(types: !26)
!26 = !{!18, !27}
!27 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!28 = !DILocalVariable(name: "i", arg: 1, scope: !24, file: !15, line: 5, type: !27)
!29 = !DILocation(line: 5, column: 15, scope: !24)
!30 = !DILocation(line: 6, column: 3, scope: !24)
!31 = distinct !DISubprogram(name: "mid", linkageName: "_Z3midll", scope: !15, file: !15, line: 9, type: !32, scopeLine: 9, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !20)
!32 = !DISubroutineType(types: !33)
!33 = !{!18, !19, !19}
!34 = !DILocalVariable(name: "l", arg: 1, scope: !31, file: !15, line: 9, type: !19)
!35 = !DILocation(line: 9, column: 16, scope: !31)
!36 = !DILocalVariable(name: "m", arg: 2, scope: !31, file: !15, line: 9, type: !19)
!37 = !DILocation(line: 9, column: 24, scope: !31)
!38 = !DILocation(line: 10, column: 10, scope: !31)
!39 = !DILocation(line: 10, column: 10, scope: !40)
!40 = !DILexicalBlockFile(scope: !31, file: !15, discriminator: 455082007)
!41 = !DILocation(line: 10, column: 3, scope: !31)
!42 = distinct !DISubprogram(name: "top", linkageName: "_Z3topl", scope: !15, file: !15, line: 13, type: !16, scopeLine: 13, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !20)
!43 = !DILocalVariable(name: "l", arg: 1, scope: !42, file: !15, line: 13, type: !19)
!44 = !DILocation(line: 13, column: 16, scope: !42)
!45 = !DILocation(line: 14, column: 14, scope: !42)
!46 = !DILocation(line: 14, column: 10, scope: !47)
!47 = !DILexicalBlockFile(scope: !42, file: !15, discriminator: 455082007)
!48 = !DILocation(line: 14, column: 3, scope: !42)
!49 = distinct !DISubprogram(name: "bar", linkageName: "_Z3barl", scope: !15, file: !15, line: 17, type: !50, scopeLine: 17, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !20)
!50 = !DISubroutineType(types: !51)
!51 = !{null, !19}
!52 = !DILocalVariable(name: "l", arg: 1, scope: !49, file: !15, line: 17, type: !19)
!53 = !DILocation(line: 17, column: 15, scope: !49)
!54 = !DILocation(line: 17, column: 19, scope: !49)
!55 = distinct !DISubprogram(name: "top", linkageName: "_Z3topll", scope: !15, file: !15, line: 19, type: !32, scopeLine: 19, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !20)
!56 = !DILocalVariable(name: "l", arg: 1, scope: !55, file: !15, line: 19, type: !19)
!57 = !DILocation(line: 19, column: 16, scope: !55)
!58 = !DILocalVariable(name: "m", arg: 2, scope: !55, file: !15, line: 19, type: !19)
!59 = !DILocation(line: 19, column: 24, scope: !55)
!60 = !DILocation(line: 20, column: 14, scope: !55)
!61 = !DILocation(line: 20, column: 17, scope: !55)
!62 = !DILocation(line: 20, column: 10, scope: !63)
!63 = !DILexicalBlockFile(scope: !55, file: !15, discriminator: 455082007)
!64 = !DILocation(line: 20, column: 3, scope: !55)
!65 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fool", scope: !15, file: !15, line: 23, type: !16, scopeLine: 23, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !20)
!66 = !DILocalVariable(name: "l", arg: 1, scope: !65, file: !15, line: 23, type: !19)
!67 = !DILocation(line: 23, column: 16, scope: !65)
!68 = !DILocation(line: 24, column: 7, scope: !69)
!69 = distinct !DILexicalBlock(scope: !65, file: !15, line: 24, column: 7)
!70 = !DILocation(line: 25, column: 16, scope: !69)
!71 = !DILocation(line: 25, column: 12, scope: !72)
!72 = !DILexicalBlockFile(scope: !69, file: !15, discriminator: 455082015)
!73 = !DILocation(line: 25, column: 5, scope: !69)
!74 = !DILocation(line: 26, column: 7, scope: !65)
!75 = !DILocation(line: 26, column: 3, scope: !76)
!76 = !DILexicalBlockFile(scope: !65, file: !15, discriminator: 455082031)
!77 = !DILocation(line: 27, column: 14, scope: !65)
!78 = !DILocation(line: 27, column: 17, scope: !65)
!79 = !DILocation(line: 27, column: 10, scope: !80)
!80 = !DILexicalBlockFile(scope: !65, file: !15, discriminator: 455082039)
!81 = !DILocation(line: 27, column: 3, scope: !65)
!82 = !DILocation(line: 28, column: 1, scope: !65)
