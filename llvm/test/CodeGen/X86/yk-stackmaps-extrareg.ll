; NOTE: Check that stackmaps can track additional registers.
; RUN: llc --yk-stackmap-spillreloads-fix --yk-stackmap-add-locs < %s | FileCheck %s

; CHECK-LABEL: __LLVM_StackMaps:
; CHECK-LABEL: .long   -56
; CHECK-NEXT: .byte 1
; CHECK-NEXT: .byte 1
; CHECK-NEXT: .byte 0
; CHECK-NEXT: .short 8
; NOTE: Actual tracked register
; CHECK-NEXT: .short 13
; NOTE: Reserved
; CHECK-NEXT: .short 0
; NOTE: Number of extra locations.
; CHECK-NEXT: .short 1
; NOTE: Stack offset this value is stored in.
; CHECK-NEXT: .short -80

source_filename = "ld-temp.o"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%YkCtrlPointVars = type { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }
%struct.YkLocation = type { i64 }

@.str = private unnamed_addr constant [14 x i8] c"out of memory\00", align 1, !dbg !0
@.str.1 = private unnamed_addr constant [2 x i8] c"+\00", align 1, !dbg !7
@shadowstack_0 = global ptr null

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !26 {
  %1 = alloca %YkCtrlPointVars, align 8
  %2 = call ptr @malloc(i64 1000000)
  call void (i64, i32, ...) @llvm.experimental.stackmap(i64 12, i32 0, ptr %1)
  store ptr %2, ptr @shadowstack_0, align 8
  %3 = getelementptr i8, ptr %2, i32 0
  %4 = getelementptr i8, ptr %2, i32 8
  %5 = getelementptr i8, ptr %2, i32 16
  %6 = alloca ptr, align 8
  %7 = getelementptr i8, ptr %2, i32 24
  %8 = alloca %struct.YkLocation, align 8
  %9 = getelementptr i8, ptr %2, i32 32
  %10 = getelementptr i8, ptr %2, i32 40
  %11 = getelementptr i8, ptr %2, i32 48
  %12 = getelementptr i8, ptr %2, i32 56
  store i32 0, ptr %3, align 4
  ; main() c/bf.c:35:9
  call void @llvm.dbg.declare(metadata ptr %4, metadata !31, metadata !DIExpression()), !dbg !33
  %13 = getelementptr i8, ptr %2, i32 64
  ; main() c/bf.c:35:17
  store ptr %13, ptr @shadowstack_0, align 8, !dbg !34
  %14 = call noalias ptr @calloc(i64 noundef 1, i64 noundef 30000) #8, !dbg !34
  ; main() c/bf.c:35:9
  ;call void (i64, i32, ...) @llvm.experimental.stackmap(i64 1, i32 0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, ptr %9, ptr %10, ptr %11, ptr %12), !dbg !33
  store ptr %2, ptr @shadowstack_0, align 8, !dbg !33
  store ptr %14, ptr %4, align 8, !dbg !33
  ; main() c/bf.c:36:7
  %15 = load ptr, ptr %4, align 8, !dbg !35
  ; main() c/bf.c:36:13
  %16 = icmp eq ptr %15, null, !dbg !37
  ; main() c/bf.c:36:7
  ;call void (i64, i32, ...) @llvm.experimental.stackmap(i64 2, i32 0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, ptr %9, ptr %10, ptr %11, ptr %12, i1 %16), !dbg !38
  br i1 %16, label %17, label %19, !dbg !38

17:                                               ; preds = %0
  %18 = getelementptr i8, ptr %2, i32 64
  ; main() c/bf.c:37:5
  store ptr %18, ptr @shadowstack_0, align 8, !dbg !39
  call void (i32, ptr, ...) @err(i32 noundef 1, ptr noundef @.str) #9, !dbg !39
  ;call void (i64, i32, ...) @llvm.experimental.stackmap(i64 3, i32 0, ptr %2), !dbg !39
  store ptr %2, ptr @shadowstack_0, align 8, !dbg !39
  unreachable, !dbg !39

19:                                               ; preds = %0
  ; main() c/bf.c:38:9
  call void @llvm.dbg.declare(metadata ptr %5, metadata !40, metadata !DIExpression()), !dbg !41
  ; main() c/bf.c:38:21
  %20 = load ptr, ptr %4, align 8, !dbg !42
  ; main() c/bf.c:38:27
  %21 = getelementptr inbounds i8, ptr %20, i64 30000, !dbg !43
  ; main() c/bf.c:38:9
  store ptr %21, ptr %5, align 8, !dbg !41
  ; main() c/bf.c:40:9
  call void @llvm.dbg.declare(metadata ptr %6, metadata !44, metadata !DIExpression()), !dbg !49
  %22 = getelementptr i8, ptr %2, i32 64
  ; main() c/bf.c:40:14
  store ptr %22, ptr @shadowstack_0, align 8, !dbg !50
  %23 = call ptr @yk_mt_new(ptr noundef null), !dbg !50
  ; main() c/bf.c:40:9
  ;call void (i64, i32, ...) @llvm.experimental.stackmap(i64 4, i32 0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, ptr %9, ptr %10, ptr %11, ptr %12), !dbg !49
  store ptr %2, ptr @shadowstack_0, align 8, !dbg !49
  store ptr %23, ptr %6, align 8, !dbg !49
  ; main() c/bf.c:41:27
  %24 = load ptr, ptr %6, align 8, !dbg !51
  %25 = getelementptr i8, ptr %2, i32 64
  ; main() c/bf.c:41:3
  store ptr %25, ptr @shadowstack_0, align 8, !dbg !52
  call void @yk_mt_hot_threshold_set(ptr noundef %24, i32 noundef 5), !dbg !52
  ; main() c/bf.c:43:10
  call void @llvm.dbg.declare(metadata ptr %7, metadata !53, metadata !DIExpression()), !dbg !57
  ;call void (i64, i32, ...) @llvm.experimental.stackmap(i64 5, i32 0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, ptr %9, ptr %10, ptr %11, ptr %12, ptr %24), !dbg !57
  store ptr %2, ptr @shadowstack_0, align 8, !dbg !57
  store i64 112, ptr %7, align 8, !dbg !57
  ; main() c/bf.c:44:14
  call void @llvm.dbg.declare(metadata ptr %8, metadata !58, metadata !DIExpression()), !dbg !65
  %26 = getelementptr i8, ptr %2, i32 64
  ; main() c/bf.c:44:20
  store ptr %26, ptr @shadowstack_0, align 8, !dbg !66
  %27 = call i64 @yk_location_new(), !dbg !66
  ;call void (i64, i32, ...) @llvm.experimental.stackmap(i64 6, i32 0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, ptr %9, ptr %10, ptr %11, ptr %12), !dbg !66
  store ptr %2, ptr @shadowstack_0, align 8, !dbg !66
  %28 = getelementptr inbounds %struct.YkLocation, ptr %8, i32 0, i32 0, !dbg !66
  store i64 %27, ptr %28, align 8, !dbg !66
  ; main() c/bf.c:46:9
  call void @llvm.dbg.declare(metadata ptr %9, metadata !67, metadata !DIExpression()), !dbg !68
  store ptr @.str.1, ptr %9, align 8, !dbg !68
  ; main() c/bf.c:47:9
  call void @llvm.dbg.declare(metadata ptr %10, metadata !69, metadata !DIExpression()), !dbg !70
  ; main() c/bf.c:47:17
  %29 = load ptr, ptr %9, align 8, !dbg !71
  ; main() c/bf.c:47:9
  store ptr %29, ptr %10, align 8, !dbg !70
  ; main() c/bf.c:48:9
  call void @llvm.dbg.declare(metadata ptr %11, metadata !72, metadata !DIExpression()), !dbg !73
  ; main() c/bf.c:48:16
  %30 = load ptr, ptr %4, align 8, !dbg !74
  ; main() c/bf.c:48:9
  store ptr %30, ptr %11, align 8, !dbg !73
  ; main() c/bf.c:49:9
  call void @llvm.dbg.declare(metadata ptr %12, metadata !75, metadata !DIExpression()), !dbg !76
  ; main() c/bf.c:49:32
  %31 = load i64, ptr %7, align 8, !dbg !77
  ; main() c/bf.c:49:21
  %32 = getelementptr inbounds [112 x i8], ptr @.str.1, i64 0, i64 %31, !dbg !78
  ; main() c/bf.c:49:9
  store ptr %32, ptr %12, align 8, !dbg !76
  ; main() c/bf.c:50:3
  br label %33, !dbg !79

33:                                               ; preds = %86, %19
  ; main() c/bf.c:50:10
  %34 = load ptr, ptr %10, align 8, !dbg !80
  ; main() c/bf.c:50:18
  %35 = load ptr, ptr %12, align 8, !dbg !81
  ; main() c/bf.c:50:16
  %36 = icmp ult ptr %34, %35, !dbg !82
  ; main() c/bf.c:50:3
  ;call void (i64, i32, ...) @llvm.experimental.stackmap(i64 7, i32 0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %8, ptr %10, ptr %11, ptr %12, i1 %36), !dbg !79
  br i1 %36, label %37, label %89, !dbg !79

37:                                               ; preds = %33
  ; main() c/bf.c:51:25
  %38 = load ptr, ptr %6, align 8, !dbg !83
  %39 = getelementptr i8, ptr %2, i32 64
  ; main() c/bf.c:51:5
  store ptr %39, ptr @shadowstack_0, align 8, !dbg !85
  %40 = getelementptr %YkCtrlPointVars, ptr %1, i32 0, i32 0, !dbg !85
  store ptr %2, ptr %40, align 8, !dbg !85
  %41 = getelementptr %YkCtrlPointVars, ptr %1, i32 0, i32 1, !dbg !85
  store ptr %3, ptr %41, align 8, !dbg !85
  %42 = getelementptr %YkCtrlPointVars, ptr %1, i32 0, i32 2, !dbg !85
  store ptr %4, ptr %42, align 8, !dbg !85
  %43 = getelementptr %YkCtrlPointVars, ptr %1, i32 0, i32 3, !dbg !85
  store ptr %5, ptr %43, align 8, !dbg !85
  %44 = getelementptr %YkCtrlPointVars, ptr %1, i32 0, i32 4, !dbg !85
  store ptr %6, ptr %44, align 8, !dbg !85
  %45 = getelementptr %YkCtrlPointVars, ptr %1, i32 0, i32 5, !dbg !85
  store ptr %8, ptr %45, align 8, !dbg !85
  %46 = getelementptr %YkCtrlPointVars, ptr %1, i32 0, i32 6, !dbg !85
  store ptr %10, ptr %46, align 8, !dbg !85
  %47 = getelementptr %YkCtrlPointVars, ptr %1, i32 0, i32 7, !dbg !85
  store ptr %11, ptr %47, align 8, !dbg !85
  %48 = getelementptr %YkCtrlPointVars, ptr %1, i32 0, i32 8, !dbg !85
  store ptr %12, ptr %48, align 8, !dbg !85
  %49 = getelementptr %YkCtrlPointVars, ptr %1, i32 0, i32 9, !dbg !85
  store ptr %38, ptr %49, align 8, !dbg !85
  %50 = call ptr @llvm.frameaddress.p0(i32 0), !dbg !85
  %51 = call ptr @__ykrt_control_point(ptr %38, ptr %8, ptr %1, ptr %50), !dbg !85
  ;call void (i64, i32, ...) @llvm.experimental.stackmap(i64 11, i32 0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %8, ptr %10, ptr %11, ptr %12, ptr %38, ptr %50), !dbg !85
  %52 = getelementptr %YkCtrlPointVars, ptr %1, i32 0, i32 0, !dbg !85
  %53 = load ptr, ptr %52, align 8, !dbg !85
  %54 = getelementptr %YkCtrlPointVars, ptr %1, i32 0, i32 1, !dbg !85
  %55 = load ptr, ptr %54, align 8, !dbg !85
  %56 = getelementptr %YkCtrlPointVars, ptr %1, i32 0, i32 2, !dbg !85
  %57 = load ptr, ptr %56, align 8, !dbg !85
  %58 = getelementptr %YkCtrlPointVars, ptr %1, i32 0, i32 3, !dbg !85
  %59 = load ptr, ptr %58, align 8, !dbg !85
  %60 = getelementptr %YkCtrlPointVars, ptr %1, i32 0, i32 4, !dbg !85
  %61 = load ptr, ptr %60, align 8, !dbg !85
  %62 = getelementptr %YkCtrlPointVars, ptr %1, i32 0, i32 5, !dbg !85
  %63 = load ptr, ptr %62, align 8, !dbg !85
  %64 = getelementptr %YkCtrlPointVars, ptr %1, i32 0, i32 6, !dbg !85
  %65 = load ptr, ptr %64, align 8, !dbg !85
  %66 = getelementptr %YkCtrlPointVars, ptr %1, i32 0, i32 7, !dbg !85
  %67 = load ptr, ptr %66, align 8, !dbg !85
  %68 = getelementptr %YkCtrlPointVars, ptr %1, i32 0, i32 8, !dbg !85
  %69 = load ptr, ptr %68, align 8, !dbg !85
  %70 = getelementptr %YkCtrlPointVars, ptr %1, i32 0, i32 9, !dbg !85
  %71 = load ptr, ptr %70, align 8, !dbg !85
  %72 = icmp eq ptr %51, null, !dbg !85
  call void (i64, i32, ...) @llvm.experimental.stackmap(i64 14, i32 0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %8, ptr %10, ptr %11, ptr %12, ptr %51, ptr %53, ptr %59, ptr %65, ptr %67, i1 %72), !dbg !85
  br i1 %72, label %73, label %95, !dbg !85

73:                                               ; preds = %37
  ; main() c/bf.c:52:14
  store ptr %53, ptr @shadowstack_0, align 8, !dbg !86
  %74 = load ptr, ptr %65, align 8, !dbg !86
  ; main() c/bf.c:52:13
  %75 = load i8, ptr %74, align 1, !dbg !87
  %76 = sext i8 %75 to i32, !dbg !87
  ; main() c/bf.c:52:5
  call void (i64, i32, ...) @llvm.experimental.stackmap(i64 8, i32 0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %8, ptr %10, ptr %11, ptr %12, ptr %53, ptr %59, ptr %65, ptr %67, i32 %76), !dbg !88
  switch i32 %76, label %85 [
    i32 62, label %77
  ], !dbg !88

77:                                               ; preds = %73
  ; main() c/bf.c:54:15
  %78 = load ptr, ptr %67, align 8, !dbg !89
  %79 = getelementptr inbounds i8, ptr %78, i32 1, !dbg !89
  store ptr %79, ptr %67, align 8, !dbg !89
  ; main() c/bf.c:54:21
  %80 = load ptr, ptr %59, align 8, !dbg !93
  ; main() c/bf.c:54:18
  %81 = icmp eq ptr %78, %80, !dbg !94
  ; main() c/bf.c:54:11
  ;call void (i64, i32, ...) @llvm.experimental.stackmap(i64 9, i32 0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %8, ptr %10, ptr %11, ptr %12, ptr %53, ptr %65, i1 %81), !dbg !95
  br i1 %81, label %82, label %84, !dbg !95

82:                                               ; preds = %77
  %83 = getelementptr i8, ptr %53, i32 64
  ; main() c/bf.c:55:9
  store ptr %83, ptr @shadowstack_0, align 8, !dbg !96
  call void (i32, ptr, ...) @errx(i32 noundef 1, ptr noundef @.str) #9, !dbg !96
  ;call void (i64, i32, ...) @llvm.experimental.stackmap(i64 10, i32 0, ptr %53), !dbg !96
  store ptr %53, ptr @shadowstack_0, align 8, !dbg !96
  unreachable, !dbg !96

84:                                               ; preds = %77
  ; main() c/bf.c:56:7
  br label %86, !dbg !97

85:                                               ; preds = %73
  ; main() c/bf.c:59:7
  br label %86, !dbg !98

86:                                               ; preds = %85, %84
  ; main() c/bf.c:61:10
  %87 = load ptr, ptr %65, align 8, !dbg !99
  %88 = getelementptr inbounds i8, ptr %87, i32 1, !dbg !99
  store ptr %88, ptr %65, align 8, !dbg !99
  ; main() c/bf.c:50:3
  br label %33, !dbg !79, !llvm.loop !100

89:                                               ; preds = %33
  ; main() c/bf.c:64:8
  %90 = load ptr, ptr %4, align 8, !dbg !103
  %91 = getelementptr i8, ptr %2, i32 64
  ; main() c/bf.c:64:3
  store ptr %91, ptr @shadowstack_0, align 8, !dbg !104
  call void @free(ptr noundef %90) #10, !dbg !104
  ; main() c/bf.c:65:1
  ;call void (i64, i32, ...) @llvm.experimental.stackmap(i64 0, i32 0, ptr %2, ptr %3, ptr %90), !dbg !105
  store ptr %2, ptr @shadowstack_0, align 8, !dbg !105
  %92 = load i32, ptr %3, align 4, !dbg !105
  br label %94, !dbg !105

93:                                               ; preds = %94
  ret i32 %92, !dbg !105

94:                                               ; preds = %89
  br label %93, !dbg !105

95:                                               ; preds = %37
  ; main() c/bf.c:51:5
  call void @__ykrt_reconstruct_frames(ptr %51), !dbg !85
  ;call void (i64, i32, ...) @llvm.experimental.stackmap(i64 13, i32 0, ptr %51), !dbg !85
  unreachable, !dbg !85
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind allocsize(0,1)
declare dso_local noalias ptr @calloc(i64 noundef, i64 noundef) #2

; Function Attrs: noreturn
declare dso_local void @err(i32 noundef, ptr noundef, ...) #3

declare dso_local ptr @yk_mt_new(ptr noundef) #4

declare dso_local void @yk_mt_hot_threshold_set(ptr noundef, i32 noundef) #4

declare dso_local i64 @yk_location_new() #4

declare dso_local void @yk_mt_control_point(ptr noundef, ptr noundef) #4

; Function Attrs: noreturn
declare dso_local void @errx(i32 noundef, ptr noundef, ...) #3

; Function Attrs: nounwind
declare dso_local void @free(ptr noundef) #5

declare ptr @malloc(i64)

declare ptr @__ykrt_control_point(ptr, ptr, ptr, ptr)

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare ptr @llvm.frameaddress.p0(i32 immarg) #6

declare void @__ykrt_reconstruct_frames(ptr)

; Function Attrs: nocallback nofree nosync willreturn
declare void @llvm.experimental.stackmap(i64, i32, ...) #7

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "yk_outline" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nounwind allocsize(0,1) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { noreturn "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #4 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #5 = { nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #6 = { nocallback nofree nosync nounwind willreturn memory(none) }
attributes #7 = { nocallback nofree nosync willreturn }
attributes #8 = { nounwind allocsize(0,1) }
attributes #9 = { noreturn }
attributes #10 = { nounwind }

!llvm.dbg.cu = !{!12}
!llvm.ident = !{!17}
!llvm.module.flags = !{!18, !19, !20, !21, !22, !23, !24, !25}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(scope: null, file: !2, line: 37, type: !3, isLocal: true, isDefinition: true)
!2 = !DIFile(filename: "c/bf.c", directory: "/home/lukasd/research/yk/tests", checksumkind: CSK_MD5, checksum: "4bd1fcf4655f08cad18c4cc8bb386c4d")
!3 = !DICompositeType(tag: DW_TAG_array_type, baseType: !4, size: 112, elements: !5)
!4 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!5 = !{!6}
!6 = !DISubrange(count: 14)
!7 = !DIGlobalVariableExpression(var: !8, expr: !DIExpression())
!8 = distinct !DIGlobalVariable(scope: null, file: !2, line: 46, type: !9, isLocal: true, isDefinition: true)
!9 = !DICompositeType(tag: DW_TAG_array_type, baseType: !4, size: 896, elements: !10)
!10 = !{!11}
!11 = !DISubrange(count: 112)
!12 = distinct !DICompileUnit(language: DW_LANG_C11, file: !13, producer: "clang version 16.0.0 (git@github.com:ptersilie/ykllvm.git bbb2156588c0846ce39def0c874b3264162b3155)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !14, globals: !16, splitDebugInlining: false, nameTableKind: None)
!13 = !DIFile(filename: "/home/lukasd/research/yk/tests/c/bf.c", directory: "/home/lukasd/research/yk/tests", checksumkind: CSK_MD5, checksum: "4bd1fcf4655f08cad18c4cc8bb386c4d")
!14 = !{!15}
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!16 = !{!0, !7}
!17 = !{!"clang version 16.0.0 (git@github.com:ptersilie/ykllvm.git bbb2156588c0846ce39def0c874b3264162b3155)"}
!18 = !{i32 7, !"Dwarf Version", i32 5}
!19 = !{i32 2, !"Debug Info Version", i32 3}
!20 = !{i32 1, !"wchar_size", i32 4}
!21 = !{i32 7, !"uwtable", i32 2}
!22 = !{i32 7, !"frame-pointer", i32 2}
!23 = !{i32 1, !"ThinLTO", i32 0}
!24 = !{i32 1, !"EnableSplitLTOUnit", i32 1}
!25 = !{i32 1, !"LTOPostLink", i32 1}
!26 = distinct !DISubprogram(name: "main", scope: !2, file: !2, line: 34, type: !27, scopeLine: 34, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !12, retainedNodes: !30)
!27 = !DISubroutineType(types: !28)
!28 = !{!29}
!29 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!30 = !{}
!31 = !DILocalVariable(name: "cells", scope: !26, file: !2, line: 35, type: !32)
!32 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !4, size: 64)
!33 = !DILocation(line: 35, column: 9, scope: !26)
!34 = !DILocation(line: 35, column: 17, scope: !26)
!35 = !DILocation(line: 36, column: 7, scope: !36)
!36 = distinct !DILexicalBlock(scope: !26, file: !2, line: 36, column: 7)
!37 = !DILocation(line: 36, column: 13, scope: !36)
!38 = !DILocation(line: 36, column: 7, scope: !26)
!39 = !DILocation(line: 37, column: 5, scope: !36)
!40 = !DILocalVariable(name: "cells_end", scope: !26, file: !2, line: 38, type: !32)
!41 = !DILocation(line: 38, column: 9, scope: !26)
!42 = !DILocation(line: 38, column: 21, scope: !26)
!43 = !DILocation(line: 38, column: 27, scope: !26)
!44 = !DILocalVariable(name: "mt", scope: !26, file: !2, line: 40, type: !45)
!45 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !46, size: 64)
!46 = !DIDerivedType(tag: DW_TAG_typedef, name: "YkMT", file: !47, line: 29, baseType: !48)
!47 = !DIFile(filename: "../ykcapi/scripts/../yk.h", directory: "/home/lukasd/research/yk/tests", checksumkind: CSK_MD5, checksum: "b5ee8232c7b9cf48d0458b57ac3b873a")
!48 = !DICompositeType(tag: DW_TAG_structure_type, name: "YkMT", file: !47, line: 29, flags: DIFlagFwdDecl)
!49 = !DILocation(line: 40, column: 9, scope: !26)
!50 = !DILocation(line: 40, column: 14, scope: !26)
!51 = !DILocation(line: 41, column: 27, scope: !26)
!52 = !DILocation(line: 41, column: 3, scope: !26)
!53 = !DILocalVariable(name: "prog_len", scope: !26, file: !2, line: 43, type: !54)
!54 = !DIDerivedType(tag: DW_TAG_typedef, name: "size_t", file: !55, line: 46, baseType: !56)
!55 = !DIFile(filename: "ykllvm/build/lib/clang/16/include/stddef.h", directory: "/home/lukasd/research", checksumkind: CSK_MD5, checksum: "f95079da609b0e8f201cb8136304bf3b")
!56 = !DIBasicType(name: "unsigned long", size: 64, encoding: DW_ATE_unsigned)
!57 = !DILocation(line: 43, column: 10, scope: !26)
!58 = !DILocalVariable(name: "loc", scope: !26, file: !2, line: 44, type: !59)
!59 = !DIDerivedType(tag: DW_TAG_typedef, name: "YkLocation", file: !47, line: 21, baseType: !60)
!60 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !47, line: 19, size: 64, elements: !61)
!61 = !{!62}
!62 = !DIDerivedType(tag: DW_TAG_member, name: "state", scope: !60, file: !47, line: 20, baseType: !63, size: 64)
!63 = !DIDerivedType(tag: DW_TAG_typedef, name: "uintptr_t", file: !64, line: 90, baseType: !56)
!64 = !DIFile(filename: "/usr/include/stdint.h", directory: "", checksumkind: CSK_MD5, checksum: "24103e292ae21916e87130b926c8d2f8")
!65 = !DILocation(line: 44, column: 14, scope: !26)
!66 = !DILocation(line: 44, column: 20, scope: !26)
!67 = !DILocalVariable(name: "prog", scope: !26, file: !2, line: 46, type: !32)
!68 = !DILocation(line: 46, column: 9, scope: !26)
!69 = !DILocalVariable(name: "instr", scope: !26, file: !2, line: 47, type: !32)
!70 = !DILocation(line: 47, column: 9, scope: !26)
!71 = !DILocation(line: 47, column: 17, scope: !26)
!72 = !DILocalVariable(name: "cell", scope: !26, file: !2, line: 48, type: !32)
!73 = !DILocation(line: 48, column: 9, scope: !26)
!74 = !DILocation(line: 48, column: 16, scope: !26)
!75 = !DILocalVariable(name: "prog_end", scope: !26, file: !2, line: 49, type: !32)
!76 = !DILocation(line: 49, column: 9, scope: !26)
!77 = !DILocation(line: 49, column: 32, scope: !26)
!78 = !DILocation(line: 49, column: 21, scope: !26)
!79 = !DILocation(line: 50, column: 3, scope: !26)
!80 = !DILocation(line: 50, column: 10, scope: !26)
!81 = !DILocation(line: 50, column: 18, scope: !26)
!82 = !DILocation(line: 50, column: 16, scope: !26)
!83 = !DILocation(line: 51, column: 25, scope: !84)
!84 = distinct !DILexicalBlock(scope: !26, file: !2, line: 50, column: 28)
!85 = !DILocation(line: 51, column: 5, scope: !84)
!86 = !DILocation(line: 52, column: 14, scope: !84)
!87 = !DILocation(line: 52, column: 13, scope: !84)
!88 = !DILocation(line: 52, column: 5, scope: !84)
!89 = !DILocation(line: 54, column: 15, scope: !90)
!90 = distinct !DILexicalBlock(scope: !91, file: !2, line: 54, column: 11)
!91 = distinct !DILexicalBlock(scope: !92, file: !2, line: 53, column: 15)
!92 = distinct !DILexicalBlock(scope: !84, file: !2, line: 52, column: 21)
!93 = !DILocation(line: 54, column: 21, scope: !90)
!94 = !DILocation(line: 54, column: 18, scope: !90)
!95 = !DILocation(line: 54, column: 11, scope: !91)
!96 = !DILocation(line: 55, column: 9, scope: !90)
!97 = !DILocation(line: 56, column: 7, scope: !91)
!98 = !DILocation(line: 59, column: 7, scope: !92)
!99 = !DILocation(line: 61, column: 10, scope: !84)
!100 = distinct !{!100, !79, !101, !102}
!101 = !DILocation(line: 62, column: 3, scope: !26)
!102 = !{!"llvm.loop.mustprogress"}
!103 = !DILocation(line: 64, column: 8, scope: !26)
!104 = !DILocation(line: 64, column: 3, scope: !26)
!105 = !DILocation(line: 65, column: 1, scope: !26)
