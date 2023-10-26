; NOTE: Check that stackmaps can track additional registers.
; RUN: llc --yk-stackmap-spillreloads-fix < %s | FileCheck %s

; CHECK-LABEL: __LLVM_StackMaps:
; CHECK-LABEL: .quad   4
; CHECK-NEXT: luaD_rawrunprotectedluaV_execute_key
; CHECK-NEXT: .short 0
; CHECK-NEXT: .short 14

; ModuleID = 'ld-temp.o'
source_filename = "ld-temp.o"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@shadowstack_0 = global ptr null

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @luaC_barrierback_(ptr noundef %0) #0 !dbg !22 {
  %2 = load ptr, ptr @shadowstack_0, align 8, !dbg !27
  %3 = getelementptr i8, ptr %2, i32 0
  store ptr %3, ptr @shadowstack_0, align 8, !dbg !27
  call void @llvm.dbg.value(metadata ptr %0, metadata !28, metadata !DIExpression()), !dbg !27
  store ptr %2, ptr @shadowstack_0, align 8, !dbg !29
  br label %5, !dbg !29

4:                                                ; preds = %5
  ret void, !dbg !29

5:                                                ; preds = %1
  br label %4, !dbg !29
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @luaF_newtbcupval(ptr noundef %0, ptr noundef %1) #0 !dbg !30 {
  %3 = load ptr, ptr @shadowstack_0, align 8, !dbg !33
  %4 = getelementptr i8, ptr %3, i32 0
  store ptr %4, ptr @shadowstack_0, align 8, !dbg !33
  call void @llvm.dbg.value(metadata ptr %0, metadata !34, metadata !DIExpression()), !dbg !33
  %5 = getelementptr i8, ptr %3, i32 0
  store ptr %5, ptr @shadowstack_0, align 8, !dbg !33
  call void @llvm.dbg.value(metadata ptr %1, metadata !35, metadata !DIExpression()), !dbg !33
  store ptr %3, ptr @shadowstack_0, align 8, !dbg !36
  store ptr %3, ptr @shadowstack_0, align 8, !dbg !36
  br label %7, !dbg !36

6:                                                ; preds = %7
  ret void, !dbg !36

7:                                                ; preds = %2
  br label %6, !dbg !36
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local ptr @luaH_getshortstr(ptr noundef %0, ptr noundef %1) #0 !dbg !37 {
  %3 = load ptr, ptr @shadowstack_0, align 8
  %4 = getelementptr i8, ptr %3, i32 0
  %5 = getelementptr i8, ptr %3, i32 16
  store ptr %5, ptr @shadowstack_0, align 8, !dbg !47
  call void @llvm.dbg.value(metadata ptr %0, metadata !48, metadata !DIExpression()), !dbg !47
  %6 = getelementptr i8, ptr %3, i32 16
  store ptr %6, ptr @shadowstack_0, align 8, !dbg !47
  call void @llvm.dbg.value(metadata ptr %1, metadata !49, metadata !DIExpression()), !dbg !47
  store ptr %3, ptr @shadowstack_0, align 8, !dbg !50
  store ptr %3, ptr @shadowstack_0, align 8, !dbg !50
  %7 = load ptr, ptr %4, align 8, !dbg !50
  br label %9, !dbg !50

8:                                                ; preds = %9
  ret ptr %7, !dbg !50

9:                                                ; preds = %2
  br label %8, !dbg !50
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @luaD_rawrunprotectedluaV_execute_key(ptr noundef %0) #2 !dbg !51 {
  %2 = load ptr, ptr @shadowstack_0, align 8
  %3 = getelementptr i8, ptr %2, i32 0
  %4 = getelementptr i8, ptr %2, i32 8
  %5 = getelementptr i8, ptr %2, i32 12
  %6 = getelementptr i8, ptr %2, i32 16
  %7 = getelementptr i8, ptr %2, i32 24
  %8 = getelementptr i8, ptr %2, i32 28
  %9 = getelementptr i8, ptr %2, i32 32
  %10 = getelementptr i8, ptr %2, i32 40
  %11 = getelementptr i8, ptr %2, i32 48
  %12 = getelementptr i8, ptr %2, i32 56
  %13 = getelementptr i8, ptr %2, i32 64
  %14 = getelementptr i8, ptr %2, i32 68
  %15 = getelementptr i8, ptr %2, i32 72
  %16 = getelementptr i8, ptr %2, i32 80
  store ptr %16, ptr @shadowstack_0, align 8, !dbg !52
  call void @llvm.dbg.value(metadata ptr %0, metadata !53, metadata !DIExpression()), !dbg !52
  store ptr %2, ptr @shadowstack_0, align 8
  store ptr %0, ptr %3, align 8
  br label %17, !dbg !54

17:                                               ; preds = %72, %1
  %18 = load i32, ptr %5, align 4, !dbg !55
  %19 = getelementptr i8, ptr %2, i32 80
  store ptr %19, ptr @shadowstack_0, align 8, !dbg !59
  call void @llvm.dbg.value(metadata i32 %18, metadata !60, metadata !DIExpression()), !dbg !59
  store ptr %2, ptr @shadowstack_0, align 8, !dbg !61
  call void (i64, i32, ...) @llvm.experimental.stackmap(i64 1, i32 0, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, ptr %9, ptr %10, ptr %11, ptr %12, ptr %13, ptr %14, ptr %15, i32 %18), !dbg !61
  switch i32 %18, label %72 [
    i32 0, label %20
  ], !dbg !61

20:                                               ; preds = %17
  %21 = getelementptr i8, ptr %2, i32 80
  store ptr %21, ptr @shadowstack_0, align 8, !dbg !62
  call void @llvm.dbg.value(metadata ptr %8, metadata !65, metadata !DIExpression()), !dbg !62
  store ptr %2, ptr @shadowstack_0, align 8, !dbg !66
  store ptr %8, ptr %9, align 8, !dbg !66
  br label %22, !dbg !67

22:                                               ; preds = %20
  %23 = getelementptr i8, ptr %2, i32 80
  store ptr %23, ptr @shadowstack_0, align 8, !dbg !68
  call void @llvm.dbg.label(metadata !69), !dbg !68
  store ptr %2, ptr @shadowstack_0, align 8, !dbg !70
  %24 = load ptr, ptr %9, align 8, !dbg !70
  %25 = getelementptr i8, ptr %2, i32 80
  store ptr %25, ptr @shadowstack_0, align 8, !dbg !62
  call void @llvm.dbg.value(metadata ptr %24, metadata !65, metadata !DIExpression()), !dbg !62
  %26 = getelementptr i8, ptr %2, i32 80
  store ptr %26, ptr @shadowstack_0, align 8, !dbg !62
  call void @llvm.dbg.value(metadata ptr %7, metadata !71, metadata !DIExpression(DW_OP_deref)), !dbg !62
  store ptr %2, ptr @shadowstack_0, align 8, !dbg !72
  store ptr %2, ptr @shadowstack_0, align 8, !dbg !72
  %27 = getelementptr i8, ptr %2, i32 80
  store ptr %27, ptr @shadowstack_0, align 8, !dbg !72
  %28 = call ptr @luaH_getshortstr(ptr noundef nonnull %7, ptr noundef %24), !dbg !72
  call void (i64, i32, ...) @llvm.experimental.stackmap(i64 0, i32 0, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, ptr %9, ptr %10, ptr %11, ptr %12, ptr %13, ptr %14, ptr %15, ptr %24)
  %29 = getelementptr i8, ptr %2, i32 80
  store ptr %29, ptr @shadowstack_0, align 8, !dbg !73
  call void @llvm.dbg.value(metadata ptr %4, metadata !74, metadata !DIExpression()), !dbg !73
  store ptr %2, ptr @shadowstack_0, align 8, !dbg !75
  store ptr %2, ptr @shadowstack_0, align 8, !dbg !75
  store ptr %4, ptr %11, align 8, !dbg !75
  %30 = load ptr, ptr %6, align 8, !dbg !76
  %31 = getelementptr i8, ptr %2, i32 80
  store ptr %31, ptr @shadowstack_0, align 8, !dbg !59
  call void @llvm.dbg.value(metadata ptr %30, metadata !78, metadata !DIExpression()), !dbg !59
  store ptr %2, ptr @shadowstack_0, align 8, !dbg !76
  %32 = icmp eq ptr %30, null, !dbg !76
  call void (i64, i32, ...) @llvm.experimental.stackmap(i64 2, i32 0, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, ptr %9, ptr %10, ptr %11, ptr %12, ptr %13, ptr %14, ptr %15, i1 %32), !dbg !79
  br i1 %32, label %34, label %33, !dbg !79

33:                                               ; preds = %22
  call void (i64, i32, ...) @llvm.experimental.stackmap(i64 3, i32 0, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, ptr %9, ptr %10, ptr %11, ptr %12, ptr %13, ptr %14, ptr %15), !dbg !76
  br i1 false, label %38, label %47, !dbg !76

34:                                               ; preds = %22
  %35 = getelementptr i8, ptr %2, i32 80
  store ptr %35, ptr @shadowstack_0, align 8, !dbg !80
  %36 = call ptr @luaH_getshortstr(ptr noundef null, ptr noundef nonnull @luaD_rawrunprotectedluaV_execute_key), !dbg !80
  call void (i64, i32, ...) @llvm.experimental.stackmap(i64 4, i32 0, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, ptr %9, ptr %10, ptr %11, ptr %12, ptr %13, ptr %14, ptr %15)
  %37 = getelementptr i8, ptr %2, i32 80
  store ptr %37, ptr @shadowstack_0, align 8, !dbg !73
  call void @llvm.dbg.value(metadata ptr %36, metadata !81, metadata !DIExpression()), !dbg !73
  store ptr %2, ptr @shadowstack_0, align 8, !dbg !82
  store ptr %2, ptr @shadowstack_0, align 8, !dbg !82
  store ptr %36, ptr %10, align 8, !dbg !82
  call void (i64, i32, ...) @llvm.experimental.stackmap(i64 5, i32 0, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, ptr %9, ptr %10, ptr %11, ptr %12, ptr %13, ptr %14, ptr %15), !dbg !79
  br i1 true, label %38, label %47, !dbg !79

38:                                               ; preds = %34, %33
  %39 = load ptr, ptr %10, align 8, !dbg !83
  %40 = getelementptr i8, ptr %2, i32 80
  store ptr %40, ptr @shadowstack_0, align 8, !dbg !73
  call void @llvm.dbg.value(metadata ptr %39, metadata !81, metadata !DIExpression()), !dbg !73
  %41 = getelementptr i8, ptr %2, i32 80
  store ptr %41, ptr @shadowstack_0, align 8, !dbg !85
  call void @llvm.dbg.value(metadata ptr %39, metadata !86, metadata !DIExpression()), !dbg !85
  call void @llvm.dbg.declare(metadata ptr %12, metadata !87, metadata !DIExpression()), !dbg !88
  store ptr %2, ptr @shadowstack_0, align 8, !dbg !89
  store ptr %2, ptr @shadowstack_0, align 8, !dbg !89
  %42 = load ptr, ptr %11, align 8, !dbg !89
  %43 = getelementptr i8, ptr %2, i32 80
  store ptr %43, ptr @shadowstack_0, align 8, !dbg !73
  call void @llvm.dbg.value(metadata ptr %42, metadata !74, metadata !DIExpression()), !dbg !73
  store ptr %2, ptr @shadowstack_0, align 8, !dbg !90
  %44 = load i64, ptr %42, align 4, !dbg !90
  store i64 %44, ptr %12, align 8, !dbg !90
  %45 = load i32, ptr %12, align 8, !dbg !91
  %46 = getelementptr i8, ptr %2, i32 80
  store ptr %46, ptr @shadowstack_0, align 8, !dbg !85
  call void @llvm.dbg.value(metadata ptr %39, metadata !86, metadata !DIExpression()), !dbg !85
  store ptr %2, ptr @shadowstack_0, align 8, !dbg !92
  store i32 %45, ptr %39, align 4, !dbg !92
  br label %47, !dbg !93

47:                                               ; preds = %38, %34, %33
  br label %48, !dbg !94

48:                                               ; preds = %47
  %49 = getelementptr i8, ptr %2, i32 80
  store ptr %49, ptr @shadowstack_0, align 8, !dbg !95
  call void @llvm.dbg.label(metadata !96), !dbg !95
  store ptr %2, ptr @shadowstack_0, align 8, !dbg !97
  %50 = load ptr, ptr %3, align 8, !dbg !97
  %51 = getelementptr i8, ptr %2, i32 80
  store ptr %51, ptr @shadowstack_0, align 8, !dbg !52
  call void @llvm.dbg.value(metadata ptr %50, metadata !53, metadata !DIExpression()), !dbg !52
  store ptr %2, ptr @shadowstack_0, align 8, !dbg !98
  %52 = load ptr, ptr %6, align 8, !dbg !98
  %53 = getelementptr i8, ptr %2, i32 80
  store ptr %53, ptr @shadowstack_0, align 8, !dbg !59
  call void @llvm.dbg.value(metadata ptr %52, metadata !78, metadata !DIExpression()), !dbg !59
  store ptr %2, ptr @shadowstack_0, align 8, !dbg !99
  %54 = getelementptr i8, ptr %2, i32 80
  store ptr %54, ptr @shadowstack_0, align 8, !dbg !99
  call void @luaF_newtbcupval(ptr noundef %50, ptr noundef %52), !dbg !99
  call void (i64, i32, ...) @llvm.experimental.stackmap(i64 6, i32 0, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, ptr %9, ptr %10, ptr %11, ptr %12, ptr %13, ptr %14, ptr %15, ptr %50, ptr %52), !dbg !100
  store ptr %2, ptr @shadowstack_0, align 8, !dbg !100
  br label %55, !dbg !100

55:                                               ; preds = %70, %48
  %56 = load i32, ptr %13, align 4, !dbg !101
  %57 = getelementptr i8, ptr %2, i32 80
  store ptr %57, ptr @shadowstack_0, align 8, !dbg !73
  call void @llvm.dbg.value(metadata i32 %56, metadata !104, metadata !DIExpression()), !dbg !73
  store ptr %2, ptr @shadowstack_0, align 8, !dbg !105
  %58 = icmp eq i32 %56, 0, !dbg !105
  call void (i64, i32, ...) @llvm.experimental.stackmap(i64 7, i32 0, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, ptr %9, ptr %10, ptr %11, ptr %12, ptr %13, ptr %14, ptr %15, i1 %58), !dbg !105
  br i1 %58, label %71, label %59, !dbg !105

59:                                               ; preds = %55
  %60 = load i8, ptr %15, align 1, !dbg !106
  %61 = getelementptr i8, ptr %2, i32 80
  store ptr %61, ptr @shadowstack_0, align 8, !dbg !108
  call void @llvm.dbg.value(metadata i8 %60, metadata !109, metadata !DIExpression()), !dbg !108
  store ptr %2, ptr @shadowstack_0, align 8, !dbg !106
  %62 = icmp eq i8 %60, 0, !dbg !106
  call void (i64, i32, ...) @llvm.experimental.stackmap(i64 8, i32 0, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, ptr %9, ptr %10, ptr %11, ptr %12, ptr %13, ptr %14, ptr %15, i1 %62), !dbg !106
  br i1 %62, label %69, label %63, !dbg !106

63:                                               ; preds = %59
  %64 = load i32, ptr %14, align 4, !dbg !110
  %65 = getelementptr i8, ptr %2, i32 80
  store ptr %65, ptr @shadowstack_0, align 8, !dbg !73
  call void @llvm.dbg.value(metadata i32 %64, metadata !111, metadata !DIExpression()), !dbg !73
  store ptr %2, ptr @shadowstack_0, align 8, !dbg !112
  %66 = sext i32 %64 to i64, !dbg !112
  %67 = inttoptr i64 %66 to ptr, !dbg !112
  %68 = getelementptr i8, ptr %2, i32 80
  store ptr %68, ptr @shadowstack_0, align 8, !dbg !113
  call void @luaC_barrierback_(ptr noundef %67), !dbg !113
  call void (i64, i32, ...) @llvm.experimental.stackmap(i64 9, i32 0, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, ptr %9, ptr %10, ptr %11, ptr %12, ptr %13, ptr %14, ptr %15, ptr %67), !dbg !106
  store ptr %2, ptr @shadowstack_0, align 8, !dbg !106
  br label %70, !dbg !106

69:                                               ; preds = %59
  br label %70, !dbg !106

70:                                               ; preds = %69, %63
  br label %55, !dbg !114, !llvm.loop !115

71:                                               ; preds = %55
  br label %72, !dbg !118

72:                                               ; preds = %71, %17
  br label %17, !dbg !119, !llvm.loop !120
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.label(metadata) #1

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #3

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @main() #0 !dbg !123 {
  %1 = call ptr @malloc(i64 1000000), !dbg !126
  call void (i64, i32, ...) @llvm.experimental.stackmap(i64 10, i32 0), !dbg !126
  store ptr %1, ptr @shadowstack_0, align 8, !dbg !126
  br label %3, !dbg !126

2:                                                ; preds = %3
  ret void, !dbg !126

3:                                                ; preds = %0
  br label %2, !dbg !126
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

declare ptr @malloc(i64)

; Function Attrs: nocallback nofree nosync willreturn
declare void @llvm.experimental.stackmap(i64, i32, ...) #4

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "yk_outline" }
attributes #3 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #4 = { nocallback nofree nosync willreturn }

!llvm.dbg.cu = !{!0}
!llvm.ident = !{!13}
!llvm.module.flags = !{!14, !15, !16, !17, !18, !19, !20, !21}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 16.0.0 (git@github.com:ykjit/ykllvm.git 2234bb62e48e3ffcd7677654ead5e72f2388f562)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !7, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "onelua.c", directory: "/home/shreei/repro_bug", checksumkind: CSK_MD5, checksum: "f38c8ed2368b352613147d524e3924f7")
!2 = !{!3}
!3 = !DICompositeType(tag: DW_TAG_enumeration_type, file: !1, line: 5, baseType: !4, size: 32, elements: !5)
!4 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!5 = !{!6}
!6 = !DIEnumerator(name: "OP_GETFIELD", value: 0)
!7 = !{!8}
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 64)
!9 = distinct !DICompositeType(tag: DW_TAG_union_type, name: "GCUnion", file: !1, line: 6, size: 32, elements: !10)
!10 = !{!11}
!11 = !DIDerivedType(tag: DW_TAG_member, name: "gc", scope: !9, file: !1, line: 7, baseType: !12, size: 32)
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !{!"clang version 16.0.0 (git@github.com:ykjit/ykllvm.git 2234bb62e48e3ffcd7677654ead5e72f2388f562)"}
!14 = !{i32 7, !"Dwarf Version", i32 5}
!15 = !{i32 2, !"Debug Info Version", i32 3}
!16 = !{i32 1, !"wchar_size", i32 4}
!17 = !{i32 7, !"uwtable", i32 2}
!18 = !{i32 7, !"frame-pointer", i32 2}
!19 = !{i32 1, !"ThinLTO", i32 0}
!20 = !{i32 1, !"EnableSplitLTOUnit", i32 1}
!21 = !{i32 1, !"LTOPostLink", i32 1}
!22 = distinct !DISubprogram(name: "luaC_barrierback_", scope: !1, file: !1, line: 9, type: !23, scopeLine: 9, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !26)
!23 = !DISubroutineType(types: !24)
!24 = !{null, !25}
!25 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!26 = !{}
!27 = !DILocation(line: 0, scope: !22)
!28 = !DILocalVariable(arg: 1, scope: !22, file: !1, line: 9, type: !25)
!29 = !DILocation(line: 9, column: 32, scope: !22)
!30 = distinct !DISubprogram(name: "luaF_newtbcupval", scope: !1, file: !1, line: 10, type: !31, scopeLine: 10, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !26)
!31 = !DISubroutineType(types: !32)
!32 = !{null, !25, !25}
!33 = !DILocation(line: 0, scope: !30)
!34 = !DILocalVariable(arg: 1, scope: !30, file: !1, line: 10, type: !25)
!35 = !DILocalVariable(arg: 2, scope: !30, file: !1, line: 10, type: !25)
!36 = !DILocation(line: 10, column: 38, scope: !30)
!37 = distinct !DISubprogram(name: "luaH_getshortstr", scope: !1, file: !1, line: 11, type: !38, scopeLine: 11, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !26)
!38 = !DISubroutineType(types: !39)
!39 = !{!40, !25, !25}
!40 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !41, size: 64)
!41 = !DIDerivedType(tag: DW_TAG_typedef, name: "TValue", file: !1, line: 4, baseType: !42)
!42 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !1, line: 1, size: 64, elements: !43)
!43 = !{!44, !45}
!44 = !DIDerivedType(tag: DW_TAG_member, name: "value_", scope: !42, file: !1, line: 2, baseType: !12, size: 32)
!45 = !DIDerivedType(tag: DW_TAG_member, name: "tt_", scope: !42, file: !1, line: 3, baseType: !46, size: 8, offset: 32)
!46 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!47 = !DILocation(line: 0, scope: !37)
!48 = !DILocalVariable(arg: 1, scope: !37, file: !1, line: 11, type: !25)
!49 = !DILocalVariable(arg: 2, scope: !37, file: !1, line: 11, type: !25)
!50 = !DILocation(line: 11, column: 41, scope: !37)
!51 = distinct !DISubprogram(name: "luaD_rawrunprotectedluaV_execute_key", scope: !1, file: !1, line: 12, type: !23, scopeLine: 12, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !26)
!52 = !DILocation(line: 0, scope: !51)
!53 = !DILocalVariable(name: "L", arg: 1, scope: !51, file: !1, line: 12, type: !25)
!54 = !DILocation(line: 14, column: 3, scope: !51)
!55 = !DILocation(line: 17, column: 13, scope: !56)
!56 = distinct !DILexicalBlock(scope: !57, file: !1, line: 14, column: 12)
!57 = distinct !DILexicalBlock(scope: !58, file: !1, line: 14, column: 3)
!58 = distinct !DILexicalBlock(scope: !51, file: !1, line: 14, column: 3)
!59 = !DILocation(line: 0, scope: !56)
!60 = !DILocalVariable(name: "i", scope: !56, file: !1, line: 15, type: !12)
!61 = !DILocation(line: 17, column: 5, scope: !56)
!62 = !DILocation(line: 0, scope: !63)
!63 = distinct !DILexicalBlock(scope: !64, file: !1, line: 18, column: 23)
!64 = distinct !DILexicalBlock(scope: !56, file: !1, line: 17, column: 16)
!65 = !DILocalVariable(name: "key", scope: !63, file: !1, line: 20, type: !25)
!66 = !DILocation(line: 20, column: 12, scope: !63)
!67 = !DILocation(line: 20, column: 7, scope: !63)
!68 = !DILocation(line: 21, column: 5, scope: !63)
!69 = !DILabel(scope: !63, name: "rb", file: !1, line: 21)
!70 = !DILocation(line: 22, column: 35, scope: !63)
!71 = !DILocalVariable(name: "rb_0_0_2", scope: !63, file: !1, line: 19, type: !12)
!72 = !DILocation(line: 22, column: 7, scope: !63)
!73 = !DILocation(line: 0, scope: !64)
!74 = !DILocalVariable(name: "rc", scope: !64, file: !1, line: 24, type: !40)
!75 = !DILocation(line: 24, column: 22, scope: !64)
!76 = !DILocation(line: 25, column: 11, scope: !77)
!77 = distinct !DILexicalBlock(scope: !64, file: !1, line: 25, column: 11)
!78 = !DILocalVariable(name: "ra", scope: !56, file: !1, line: 16, type: !25)
!79 = !DILocation(line: 25, column: 11, scope: !64)
!80 = !DILocation(line: 27, column: 21, scope: !77)
!81 = !DILocalVariable(name: "slot", scope: !64, file: !1, line: 24, type: !40)
!82 = !DILocation(line: 26, column: 22, scope: !77)
!83 = !DILocation(line: 29, column: 23, scope: !84)
!84 = distinct !DILexicalBlock(scope: !77, file: !1, line: 28, column: 22)
!85 = !DILocation(line: 0, scope: !84)
!86 = !DILocalVariable(name: "io1", scope: !84, file: !1, line: 29, type: !40)
!87 = !DILocalVariable(name: "io2", scope: !84, file: !1, line: 30, type: !41)
!88 = !DILocation(line: 30, column: 16, scope: !84)
!89 = !DILocation(line: 30, column: 23, scope: !84)
!90 = !DILocation(line: 30, column: 22, scope: !84)
!91 = !DILocation(line: 31, column: 27, scope: !84)
!92 = !DILocation(line: 31, column: 21, scope: !84)
!93 = !DILocation(line: 32, column: 7, scope: !84)
!94 = !DILocation(line: 28, column: 19, scope: !77)
!95 = !DILocation(line: 33, column: 5, scope: !64)
!96 = !DILabel(scope: !64, name: "OP_TFORPREP", file: !1, line: 33)
!97 = !DILocation(line: 34, column: 24, scope: !64)
!98 = !DILocation(line: 34, column: 27, scope: !64)
!99 = !DILocation(line: 34, column: 7, scope: !64)
!100 = !DILocation(line: 36, column: 7, scope: !64)
!101 = !DILocation(line: 36, column: 14, scope: !102)
!102 = distinct !DILexicalBlock(scope: !103, file: !1, line: 36, column: 7)
!103 = distinct !DILexicalBlock(scope: !64, file: !1, line: 36, column: 7)
!104 = !DILocalVariable(name: "n", scope: !64, file: !1, line: 35, type: !12)
!105 = !DILocation(line: 36, column: 7, scope: !103)
!106 = !DILocation(line: 38, column: 9, scope: !107)
!107 = distinct !DILexicalBlock(scope: !102, file: !1, line: 36, column: 18)
!108 = !DILocation(line: 0, scope: !107)
!109 = !DILocalVariable(name: "val_0_0_0", scope: !107, file: !1, line: 37, type: !46)
!110 = !DILocation(line: 38, column: 58, scope: !107)
!111 = !DILocalVariable(name: "h", scope: !64, file: !1, line: 35, type: !12)
!112 = !DILocation(line: 38, column: 41, scope: !107)
!113 = !DILocation(line: 38, column: 21, scope: !107)
!114 = !DILocation(line: 36, column: 7, scope: !102)
!115 = distinct !{!115, !105, !116, !117}
!116 = !DILocation(line: 39, column: 7, scope: !103)
!117 = !{!"llvm.loop.mustprogress"}
!118 = !DILocation(line: 40, column: 5, scope: !64)
!119 = !DILocation(line: 14, column: 3, scope: !57)
!120 = distinct !{!120, !121, !122}
!121 = !DILocation(line: 14, column: 3, scope: !58)
!122 = !DILocation(line: 41, column: 3, scope: !58)
!123 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 43, type: !124, scopeLine: 43, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !26)
!124 = !DISubroutineType(types: !125)
!125 = !{null}
!126 = !DILocation(line: 43, column: 14, scope: !123)
