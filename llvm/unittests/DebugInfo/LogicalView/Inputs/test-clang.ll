; ModuleID = 'test.cpp'
source_filename = "test.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux"

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local noundef i32 @_Z3fooPKijb(ptr noundef %ParamPtr, i32 noundef %ParamUnsigned, i1 noundef zeroext %ParamBool) #0 !dbg !10 {
entry:
  %retval = alloca i32, align 4
  %ParamPtr.addr = alloca ptr, align 8
  %ParamUnsigned.addr = alloca i32, align 4
  %ParamBool.addr = alloca i8, align 1
  %CONSTANT = alloca i32, align 4
  store ptr %ParamPtr, ptr %ParamPtr.addr, align 8
    #dbg_declare(ptr %ParamPtr.addr, !20, !DIExpression(), !21)
  store i32 %ParamUnsigned, ptr %ParamUnsigned.addr, align 4
    #dbg_declare(ptr %ParamUnsigned.addr, !22, !DIExpression(), !23)
  %storedv = zext i1 %ParamBool to i8
  store i8 %storedv, ptr %ParamBool.addr, align 1
    #dbg_declare(ptr %ParamBool.addr, !24, !DIExpression(), !25)
  %0 = load i8, ptr %ParamBool.addr, align 1, !dbg !26
  %loadedv = trunc i8 %0 to i1, !dbg !26
  br i1 %loadedv, label %if.then, label %if.end, !dbg !26

if.then:                                          ; preds = %entry
    #dbg_declare(ptr %CONSTANT, !28, !DIExpression(), !32)
  store i32 7, ptr %CONSTANT, align 4, !dbg !32
  store i32 7, ptr %retval, align 4, !dbg !33
  br label %return, !dbg !33

if.end:                                           ; preds = %entry
  %1 = load i32, ptr %ParamUnsigned.addr, align 4, !dbg !34
  store i32 %1, ptr %retval, align 4, !dbg !35
  br label %return, !dbg !35

return:                                           ; preds = %if.end, %if.then
  %2 = load i32, ptr %retval, align 4, !dbg !36
  ret i32 %2, !dbg !36
}

attributes #0 = { mustprogress noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 20.0.0git (https://github.com/llvm/llvm-project.git 16e45b8fac797c6d4ba161228b54665492204a9d)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "X:\\scripts\\regression-suite\\input\\general", checksumkind: CSK_MD5, checksum: "969bd5763e7769d696eb29bdc55331d7")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"frame-pointer", i32 2}
!9 = !{!"clang version 20.0.0git (https://github.com/llvm/llvm-project.git 16e45b8fac797c6d4ba161228b54665492204a9d)"}
!10 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fooPKijb", scope: !1, file: !1, line: 2, type: !11, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !19)
!11 = !DISubroutineType(types: !12)
!12 = !{!13, !14, !17, !18}
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !DIDerivedType(tag: DW_TAG_typedef, name: "INTPTR", file: !1, line: 1, baseType: !15)
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !16, size: 64)
!16 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !13)
!17 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!18 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!19 = !{}
!20 = !DILocalVariable(name: "ParamPtr", arg: 1, scope: !10, file: !1, line: 2, type: !14)
!21 = !DILocation(line: 2, column: 16, scope: !10)
!22 = !DILocalVariable(name: "ParamUnsigned", arg: 2, scope: !10, file: !1, line: 2, type: !17)
!23 = !DILocation(line: 2, column: 35, scope: !10)
!24 = !DILocalVariable(name: "ParamBool", arg: 3, scope: !10, file: !1, line: 2, type: !18)
!25 = !DILocation(line: 2, column: 55, scope: !10)
!26 = !DILocation(line: 3, column: 7, scope: !27)
!27 = distinct !DILexicalBlock(scope: !10, file: !1, line: 3, column: 7)
!28 = !DILocalVariable(name: "CONSTANT", scope: !29, file: !1, line: 5, type: !30)
!29 = distinct !DILexicalBlock(scope: !27, file: !1, line: 3, column: 18)
!30 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !31)
!31 = !DIDerivedType(tag: DW_TAG_typedef, name: "INTEGER", scope: !10, file: !1, line: 4, baseType: !13)
!32 = !DILocation(line: 5, column: 19, scope: !29)
!33 = !DILocation(line: 6, column: 5, scope: !29)
!34 = !DILocation(line: 8, column: 10, scope: !10)
!35 = !DILocation(line: 8, column: 3, scope: !10)
!36 = !DILocation(line: 9, column: 1, scope: !10)
