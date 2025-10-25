; ModuleID = 'typedef.cpp'
source_filename = "typedef.cpp"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.44.35214"

; Function Attrs: mustprogress noinline norecurse nounwind optnone uwtable
define dso_local noundef i32 @main() #0 !dbg !14 {
entry:
  %retval = alloca i32, align 4
  %val = alloca i8, align 1
  %val2 = alloca i64, align 8
  store i32 0, ptr %retval, align 4
    #dbg_declare(ptr %val, !19, !DIExpression(), !22)
  store i8 15, ptr %val, align 1, !dbg !22
    #dbg_declare(ptr %val2, !23, !DIExpression(), !26)
  store i64 -1, ptr %val2, align 8, !dbg !26
  ret i32 0, !dbg !27
}

attributes #0 = { mustprogress noinline norecurse nounwind optnone uwtable }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8}
!llvm.ident = !{!13}

!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !5, producer: "clang version 22.0.0git (https://github.com/Walnut356/llvm-project.git 0e257f7d7edfcda5655eeaac55d0ffc398e773a2)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!5 = !DIFile(filename: "typedef.cpp", directory: "llvm-project")
!7 = !{i32 2, !"CodeView", i32 1}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !{!"clang version 22.0.0git (https://github.com/Walnut356/llvm-project.git 0e257f7d7edfcda5655eeaac55d0ffc398e773a2)"}
!14 = distinct !DISubprogram(name: "main", scope: !5, file: !5, line: 6, type: !15, scopeLine: 6, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2)
!15 = !DISubroutineType(types: !16)
!16 = !{!17}
!17 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!19 = !DILocalVariable(name: "val", scope: !14, file: !5, line: 7, type: !20)
!20 = !DIDerivedType(tag: DW_TAG_typedef, name: "u8", file: !5, line: 3, baseType: !21)
!21 = !DIBasicType(name: "unsigned char", size: 8, encoding: DW_ATE_unsigned_char)
!22 = !DILocation(line: 7, scope: !14)
!23 = !DILocalVariable(name: "val2", scope: !14, file: !5, line: 8, type: !24)
!24 = !DIDerivedType(tag: DW_TAG_typedef, name: "i64", file: !5, line: 4, baseType: !25)
!25 = !DIBasicType(name: "long long", size: 64, encoding: DW_ATE_signed)
!26 = !DILocation(line: 8, scope: !14)
!27 = !DILocation(line: 10, scope: !14)
