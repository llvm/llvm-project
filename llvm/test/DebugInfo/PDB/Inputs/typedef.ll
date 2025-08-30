; ModuleID = 'typedef.cpp'
source_filename = "typedef.cpp"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.44.35214"

@"?__purecall@@3PEAXEA" = dso_local global ptr null, align 8, !dbg !0

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

attributes #0 = { mustprogress noinline norecurse nounwind optnone uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9, !10, !11, !12}
!llvm.ident = !{!13}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "__purecall", linkageName: "?__purecall@@3PEAXEA", scope: !2, file: !5, line: 1, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 22.0.0git (https://github.com/Walnut356/llvm-project.git 0e257f7d7edfcda5655eeaac55d0ffc398e773a2)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "typedef.cpp", directory: "llvm-project", checksumkind: CSK_MD5, checksum: "8c85f4e9ba063d42c5a4ad392521faa3")
!4 = !{!0}
!5 = !DIFile(filename: "typedef.cpp", directory: "llvm-project", checksumkind: CSK_MD5, checksum: "8c85f4e9ba063d42c5a4ad392521faa3")
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!7 = !{i32 2, !"CodeView", i32 1}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 2}
!10 = !{i32 8, !"PIC Level", i32 2}
!11 = !{i32 7, !"uwtable", i32 2}
!12 = !{i32 1, !"MaxTLSAlign", i32 65536}
!13 = !{!"clang version 22.0.0git (https://github.com/Walnut356/llvm-project.git 0e257f7d7edfcda5655eeaac55d0ffc398e773a2)"}
!14 = distinct !DISubprogram(name: "main", scope: !5, file: !5, line: 6, type: !15, scopeLine: 6, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !18)
!15 = !DISubroutineType(types: !16)
!16 = !{!17}
!17 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!18 = !{}
!19 = !DILocalVariable(name: "val", scope: !14, file: !5, line: 7, type: !20)
!20 = !DIDerivedType(tag: DW_TAG_typedef, name: "u8", file: !5, line: 3, baseType: !21)
!21 = !DIBasicType(name: "unsigned char", size: 8, encoding: DW_ATE_unsigned_char)
!22 = !DILocation(line: 7, scope: !14)
!23 = !DILocalVariable(name: "val2", scope: !14, file: !5, line: 8, type: !24)
!24 = !DIDerivedType(tag: DW_TAG_typedef, name: "i64", file: !5, line: 4, baseType: !25)
!25 = !DIBasicType(name: "long long", size: 64, encoding: DW_ATE_signed)
!26 = !DILocation(line: 8, scope: !14)
!27 = !DILocation(line: 10, scope: !14)
