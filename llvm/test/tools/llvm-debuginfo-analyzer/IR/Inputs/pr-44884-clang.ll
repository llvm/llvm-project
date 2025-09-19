; ModuleID = 'pr-44884.cpp'
source_filename = "pr-44884.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux"

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local noundef i32 @_Z3barf(float noundef %Input) #0 !dbg !12 {
entry:
  %Input.addr = alloca float, align 4
  store float %Input, ptr %Input.addr, align 4
    #dbg_declare(ptr %Input.addr, !17, !DIExpression(), !18)
  %0 = load float, ptr %Input.addr, align 4, !dbg !19
  %conv = fptosi float %0 to i32, !dbg !19
  ret i32 %conv, !dbg !20
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local noundef i32 @_Z3fooc(i8 noundef signext %Param) #0 !dbg !21 {
entry:
  %Param.addr = alloca i8, align 1
  %Value = alloca i32, align 4
  %Added = alloca float, align 4
  store i8 %Param, ptr %Param.addr, align 1
    #dbg_declare(ptr %Param.addr, !26, !DIExpression(), !27)
    #dbg_declare(ptr %Value, !28, !DIExpression(), !30)
  %0 = load i8, ptr %Param.addr, align 1, !dbg !31
  %conv = sext i8 %0 to i32, !dbg !31
  store i32 %conv, ptr %Value, align 4, !dbg !30
    #dbg_declare(ptr %Added, !32, !DIExpression(), !36)
  %1 = load i32, ptr %Value, align 4, !dbg !37
  %2 = load i8, ptr %Param.addr, align 1, !dbg !38
  %conv1 = sext i8 %2 to i32, !dbg !38
  %add = add nsw i32 %1, %conv1, !dbg !39
  %conv2 = sitofp i32 %add to float, !dbg !37
  store float %conv2, ptr %Added, align 4, !dbg !36
  %3 = load float, ptr %Added, align 4, !dbg !40
  %call = call noundef i32 @_Z3barf(float noundef %3), !dbg !41
  store i32 %call, ptr %Value, align 4, !dbg !42
  %4 = load i32, ptr %Value, align 4, !dbg !43
  %5 = load i8, ptr %Param.addr, align 1, !dbg !44
  %conv3 = sext i8 %5 to i32, !dbg !44
  %add4 = add nsw i32 %4, %conv3, !dbg !45
  ret i32 %add4, !dbg !46
}

attributes #0 = { mustprogress noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!4, !5, !6, !7, !8, !9, !10}
!llvm.ident = !{!11}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 20.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "pr-44884.cpp", directory: "/data/projects/scripts/regression-suite/input/general")
!2 = !{!3}
!3 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!4 = !{i32 7, !"Dwarf Version", i32 5}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = !{i32 1, !"wchar_size", i32 4}
!7 = !{i32 8, !"PIC Level", i32 2}
!8 = !{i32 7, !"PIE Level", i32 2}
!9 = !{i32 7, !"uwtable", i32 2}
!10 = !{i32 7, !"frame-pointer", i32 2}
!11 = !{!"clang version 20.0.0"}
!12 = distinct !DISubprogram(name: "bar", linkageName: "_Z3barf", scope: !1, file: !1, line: 1, type: !13, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !16)
!13 = !DISubroutineType(types: !14)
!14 = !{!3, !15}
!15 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!16 = !{}
!17 = !DILocalVariable(name: "Input", arg: 1, scope: !12, file: !1, line: 1, type: !15)
!18 = !DILocation(line: 1, column: 15, scope: !12)
!19 = !DILocation(line: 1, column: 36, scope: !12)
!20 = !DILocation(line: 1, column: 24, scope: !12)
!21 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fooc", scope: !1, file: !1, line: 3, type: !22, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !16)
!22 = !DISubroutineType(types: !23)
!23 = !{!24, !25}
!24 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!25 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!26 = !DILocalVariable(name: "Param", arg: 1, scope: !21, file: !1, line: 3, type: !25)
!27 = !DILocation(line: 3, column: 19, scope: !21)
!28 = !DILocalVariable(name: "Value", scope: !21, file: !1, line: 5, type: !29)
!29 = !DIDerivedType(tag: DW_TAG_typedef, name: "INT", scope: !21, file: !1, line: 4, baseType: !3)
!30 = !DILocation(line: 5, column: 7, scope: !21)
!31 = !DILocation(line: 5, column: 15, scope: !21)
!32 = !DILocalVariable(name: "Added", scope: !33, file: !1, line: 9, type: !35)
!33 = distinct !DILexicalBlock(scope: !34, file: !1, line: 8, column: 5)
!34 = distinct !DILexicalBlock(scope: !21, file: !1, line: 6, column: 3)
!35 = !DIDerivedType(tag: DW_TAG_typedef, name: "FLOAT", scope: !21, file: !1, line: 7, baseType: !15)
!36 = !DILocation(line: 9, column: 13, scope: !33)
!37 = !DILocation(line: 9, column: 21, scope: !33)
!38 = !DILocation(line: 9, column: 29, scope: !33)
!39 = !DILocation(line: 9, column: 27, scope: !33)
!40 = !DILocation(line: 10, column: 19, scope: !33)
!41 = !DILocation(line: 10, column: 15, scope: !33)
!42 = !DILocation(line: 10, column: 13, scope: !33)
!43 = !DILocation(line: 13, column: 10, scope: !21)
!44 = !DILocation(line: 13, column: 18, scope: !21)
!45 = !DILocation(line: 13, column: 16, scope: !21)
!46 = !DILocation(line: 13, column: 3, scope: !21)
