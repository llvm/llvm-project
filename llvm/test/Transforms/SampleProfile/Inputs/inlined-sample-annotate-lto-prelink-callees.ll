; ModuleID = 'src/heavy.cpp'
source_filename = "src/heavy.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-redhat-linux-gnu"

$_Z5heavyIN3lib2v23TagEEii = comdat any

; Function Attrs: mustprogress nounwind uwtable
define weak_odr dso_local noundef i32 @_Z5heavyIN3lib2v23TagEEii(i32 noundef %n) #0 comdat !dbg !11 {
entry:
  %n.addr = alloca i32, align 4
  %r = alloca i32, align 4
  store i32 %n, ptr %n.addr, align 4, !tbaa !24
    #dbg_declare(ptr %n.addr, !16, !DIExpression(), !25)
  call void @llvm.lifetime.start.p0(ptr %r) #2, !dbg !26
    #dbg_declare(ptr %r, !17, !DIExpression(), !27)
  %0 = load i32, ptr %n.addr, align 4, !dbg !28, !tbaa !24
  store i32 %0, ptr %r, align 4, !dbg !29, !tbaa !24
  %1 = load i32, ptr %r, align 4, !dbg !30, !tbaa !24
  %2 = load i32, ptr %n.addr, align 4, !dbg !30, !tbaa !24
  %xor = xor i32 %1, %2, !dbg !30
  %mul = mul nsw i32 %xor, 33, !dbg !31
  store i32 %mul, ptr %r, align 4, !dbg !32, !tbaa !24
  %3 = load i32, ptr %r, align 4, !dbg !30, !tbaa !24
  %4 = load i32, ptr %n.addr, align 4, !dbg !30, !tbaa !24
  %xor1 = xor i32 %3, %4, !dbg !30
  %mul2 = mul nsw i32 %xor1, 33, !dbg !33
  store i32 %mul2, ptr %r, align 4, !dbg !34, !tbaa !24
  %5 = load i32, ptr %r, align 4, !dbg !30, !tbaa !24
  %6 = load i32, ptr %n.addr, align 4, !dbg !30, !tbaa !24
  %xor3 = xor i32 %5, %6, !dbg !30
  %mul4 = mul nsw i32 %xor3, 33, !dbg !35
  store i32 %mul4, ptr %r, align 4, !dbg !36, !tbaa !24
  %7 = load i32, ptr %r, align 4, !dbg !30, !tbaa !24
  %8 = load i32, ptr %n.addr, align 4, !dbg !30, !tbaa !24
  %xor5 = xor i32 %7, %8, !dbg !30
  %mul6 = mul nsw i32 %xor5, 33, !dbg !37
  store i32 %mul6, ptr %r, align 4, !dbg !38, !tbaa !24
  %9 = load i32, ptr %r, align 4, !dbg !39, !tbaa !24
  %10 = load i32, ptr %n.addr, align 4, !dbg !39, !tbaa !24
  %xor7 = xor i32 %9, %10, !dbg !39
  %mul8 = mul nsw i32 %xor7, 33, !dbg !40
  store i32 %mul8, ptr %r, align 4, !dbg !41, !tbaa !24
  %11 = load i32, ptr %r, align 4, !dbg !39, !tbaa !24
  %12 = load i32, ptr %n.addr, align 4, !dbg !39, !tbaa !24
  %xor9 = xor i32 %11, %12, !dbg !39
  %mul10 = mul nsw i32 %xor9, 33, !dbg !42
  store i32 %mul10, ptr %r, align 4, !dbg !43, !tbaa !24
  %13 = load i32, ptr %r, align 4, !dbg !39, !tbaa !24
  %14 = load i32, ptr %n.addr, align 4, !dbg !39, !tbaa !24
  %xor11 = xor i32 %13, %14, !dbg !39
  %mul12 = mul nsw i32 %xor11, 33, !dbg !44
  store i32 %mul12, ptr %r, align 4, !dbg !45, !tbaa !24
  %15 = load i32, ptr %r, align 4, !dbg !39, !tbaa !24
  %16 = load i32, ptr %n.addr, align 4, !dbg !39, !tbaa !24
  %xor13 = xor i32 %15, %16, !dbg !39
  %mul14 = mul nsw i32 %xor13, 33, !dbg !46
  store i32 %mul14, ptr %r, align 4, !dbg !47, !tbaa !24
  %17 = load i32, ptr %r, align 4, !dbg !48, !tbaa !24
  %18 = load i32, ptr %n.addr, align 4, !dbg !48, !tbaa !24
  %xor15 = xor i32 %17, %18, !dbg !48
  %mul16 = mul nsw i32 %xor15, 33, !dbg !49
  store i32 %mul16, ptr %r, align 4, !dbg !50, !tbaa !24
  %19 = load i32, ptr %r, align 4, !dbg !48, !tbaa !24
  %20 = load i32, ptr %n.addr, align 4, !dbg !48, !tbaa !24
  %xor17 = xor i32 %19, %20, !dbg !48
  %mul18 = mul nsw i32 %xor17, 33, !dbg !51
  store i32 %mul18, ptr %r, align 4, !dbg !52, !tbaa !24
  %21 = load i32, ptr %r, align 4, !dbg !48, !tbaa !24
  %22 = load i32, ptr %n.addr, align 4, !dbg !48, !tbaa !24
  %xor19 = xor i32 %21, %22, !dbg !48
  %mul20 = mul nsw i32 %xor19, 33, !dbg !53
  store i32 %mul20, ptr %r, align 4, !dbg !54, !tbaa !24
  %23 = load i32, ptr %r, align 4, !dbg !48, !tbaa !24
  %24 = load i32, ptr %n.addr, align 4, !dbg !48, !tbaa !24
  %xor21 = xor i32 %23, %24, !dbg !48
  %mul22 = mul nsw i32 %xor21, 33, !dbg !55
  store i32 %mul22, ptr %r, align 4, !dbg !56, !tbaa !24
  %25 = load i32, ptr %r, align 4, !dbg !57, !tbaa !24
  %26 = load i32, ptr %n.addr, align 4, !dbg !57, !tbaa !24
  %xor23 = xor i32 %25, %26, !dbg !57
  %mul24 = mul nsw i32 %xor23, 33, !dbg !58
  store i32 %mul24, ptr %r, align 4, !dbg !59, !tbaa !24
  %27 = load i32, ptr %r, align 4, !dbg !57, !tbaa !24
  %28 = load i32, ptr %n.addr, align 4, !dbg !57, !tbaa !24
  %xor25 = xor i32 %27, %28, !dbg !57
  %mul26 = mul nsw i32 %xor25, 33, !dbg !60
  store i32 %mul26, ptr %r, align 4, !dbg !61, !tbaa !24
  %29 = load i32, ptr %r, align 4, !dbg !57, !tbaa !24
  %30 = load i32, ptr %n.addr, align 4, !dbg !57, !tbaa !24
  %xor27 = xor i32 %29, %30, !dbg !57
  %mul28 = mul nsw i32 %xor27, 33, !dbg !62
  store i32 %mul28, ptr %r, align 4, !dbg !63, !tbaa !24
  %31 = load i32, ptr %r, align 4, !dbg !57, !tbaa !24
  %32 = load i32, ptr %n.addr, align 4, !dbg !57, !tbaa !24
  %xor29 = xor i32 %31, %32, !dbg !57
  %mul30 = mul nsw i32 %xor29, 33, !dbg !64
  store i32 %mul30, ptr %r, align 4, !dbg !65, !tbaa !24
  %33 = load i32, ptr %r, align 4, !dbg !66, !tbaa !24
  %34 = load i32, ptr %n.addr, align 4, !dbg !66, !tbaa !24
  %xor31 = xor i32 %33, %34, !dbg !66
  %mul32 = mul nsw i32 %xor31, 33, !dbg !67
  store i32 %mul32, ptr %r, align 4, !dbg !68, !tbaa !24
  %35 = load i32, ptr %r, align 4, !dbg !66, !tbaa !24
  %36 = load i32, ptr %n.addr, align 4, !dbg !66, !tbaa !24
  %xor33 = xor i32 %35, %36, !dbg !66
  %mul34 = mul nsw i32 %xor33, 33, !dbg !69
  store i32 %mul34, ptr %r, align 4, !dbg !70, !tbaa !24
  %37 = load i32, ptr %r, align 4, !dbg !66, !tbaa !24
  %38 = load i32, ptr %n.addr, align 4, !dbg !66, !tbaa !24
  %xor35 = xor i32 %37, %38, !dbg !66
  %mul36 = mul nsw i32 %xor35, 33, !dbg !71
  store i32 %mul36, ptr %r, align 4, !dbg !72, !tbaa !24
  %39 = load i32, ptr %r, align 4, !dbg !66, !tbaa !24
  %40 = load i32, ptr %n.addr, align 4, !dbg !66, !tbaa !24
  %xor37 = xor i32 %39, %40, !dbg !66
  %mul38 = mul nsw i32 %xor37, 33, !dbg !73
  store i32 %mul38, ptr %r, align 4, !dbg !74, !tbaa !24
  %41 = load i32, ptr %r, align 4, !dbg !75, !tbaa !24
  %42 = load i32, ptr %n.addr, align 4, !dbg !75, !tbaa !24
  %xor39 = xor i32 %41, %42, !dbg !75
  %mul40 = mul nsw i32 %xor39, 33, !dbg !76
  store i32 %mul40, ptr %r, align 4, !dbg !77, !tbaa !24
  %43 = load i32, ptr %r, align 4, !dbg !75, !tbaa !24
  %44 = load i32, ptr %n.addr, align 4, !dbg !75, !tbaa !24
  %xor41 = xor i32 %43, %44, !dbg !75
  %mul42 = mul nsw i32 %xor41, 33, !dbg !78
  store i32 %mul42, ptr %r, align 4, !dbg !79, !tbaa !24
  %45 = load i32, ptr %r, align 4, !dbg !75, !tbaa !24
  %46 = load i32, ptr %n.addr, align 4, !dbg !75, !tbaa !24
  %xor43 = xor i32 %45, %46, !dbg !75
  %mul44 = mul nsw i32 %xor43, 33, !dbg !80
  store i32 %mul44, ptr %r, align 4, !dbg !81, !tbaa !24
  %47 = load i32, ptr %r, align 4, !dbg !75, !tbaa !24
  %48 = load i32, ptr %n.addr, align 4, !dbg !75, !tbaa !24
  %xor45 = xor i32 %47, %48, !dbg !75
  %mul46 = mul nsw i32 %xor45, 33, !dbg !82
  store i32 %mul46, ptr %r, align 4, !dbg !83, !tbaa !24
  %49 = load i32, ptr %r, align 4, !dbg !84, !tbaa !24
  %50 = load i32, ptr %n.addr, align 4, !dbg !84, !tbaa !24
  %xor47 = xor i32 %49, %50, !dbg !84
  %mul48 = mul nsw i32 %xor47, 33, !dbg !85
  store i32 %mul48, ptr %r, align 4, !dbg !86, !tbaa !24
  %51 = load i32, ptr %r, align 4, !dbg !84, !tbaa !24
  %52 = load i32, ptr %n.addr, align 4, !dbg !84, !tbaa !24
  %xor49 = xor i32 %51, %52, !dbg !84
  %mul50 = mul nsw i32 %xor49, 33, !dbg !87
  store i32 %mul50, ptr %r, align 4, !dbg !88, !tbaa !24
  %53 = load i32, ptr %r, align 4, !dbg !84, !tbaa !24
  %54 = load i32, ptr %n.addr, align 4, !dbg !84, !tbaa !24
  %xor51 = xor i32 %53, %54, !dbg !84
  %mul52 = mul nsw i32 %xor51, 33, !dbg !89
  store i32 %mul52, ptr %r, align 4, !dbg !90, !tbaa !24
  %55 = load i32, ptr %r, align 4, !dbg !84, !tbaa !24
  %56 = load i32, ptr %n.addr, align 4, !dbg !84, !tbaa !24
  %xor53 = xor i32 %55, %56, !dbg !84
  %mul54 = mul nsw i32 %xor53, 33, !dbg !91
  store i32 %mul54, ptr %r, align 4, !dbg !92, !tbaa !24
  %57 = load i32, ptr %r, align 4, !dbg !93, !tbaa !24
  %58 = load i32, ptr %n.addr, align 4, !dbg !93, !tbaa !24
  %xor55 = xor i32 %57, %58, !dbg !93
  %mul56 = mul nsw i32 %xor55, 33, !dbg !94
  store i32 %mul56, ptr %r, align 4, !dbg !95, !tbaa !24
  %59 = load i32, ptr %r, align 4, !dbg !93, !tbaa !24
  %60 = load i32, ptr %n.addr, align 4, !dbg !93, !tbaa !24
  %xor57 = xor i32 %59, %60, !dbg !93
  %mul58 = mul nsw i32 %xor57, 33, !dbg !96
  store i32 %mul58, ptr %r, align 4, !dbg !97, !tbaa !24
  %61 = load i32, ptr %r, align 4, !dbg !93, !tbaa !24
  %62 = load i32, ptr %n.addr, align 4, !dbg !93, !tbaa !24
  %xor59 = xor i32 %61, %62, !dbg !93
  %mul60 = mul nsw i32 %xor59, 33, !dbg !98
  store i32 %mul60, ptr %r, align 4, !dbg !99, !tbaa !24
  %63 = load i32, ptr %r, align 4, !dbg !93, !tbaa !24
  %64 = load i32, ptr %n.addr, align 4, !dbg !93, !tbaa !24
  %xor61 = xor i32 %63, %64, !dbg !93
  %mul62 = mul nsw i32 %xor61, 33, !dbg !100
  store i32 %mul62, ptr %r, align 4, !dbg !101, !tbaa !24
  %65 = load i32, ptr %r, align 4, !dbg !102, !tbaa !24
  %66 = load i32, ptr %n.addr, align 4, !dbg !102, !tbaa !24
  %xor63 = xor i32 %65, %66, !dbg !102
  %mul64 = mul nsw i32 %xor63, 33, !dbg !103
  store i32 %mul64, ptr %r, align 4, !dbg !104, !tbaa !24
  %67 = load i32, ptr %r, align 4, !dbg !102, !tbaa !24
  %68 = load i32, ptr %n.addr, align 4, !dbg !102, !tbaa !24
  %xor65 = xor i32 %67, %68, !dbg !102
  %mul66 = mul nsw i32 %xor65, 33, !dbg !105
  store i32 %mul66, ptr %r, align 4, !dbg !106, !tbaa !24
  %69 = load i32, ptr %r, align 4, !dbg !102, !tbaa !24
  %70 = load i32, ptr %n.addr, align 4, !dbg !102, !tbaa !24
  %xor67 = xor i32 %69, %70, !dbg !102
  %mul68 = mul nsw i32 %xor67, 33, !dbg !107
  store i32 %mul68, ptr %r, align 4, !dbg !108, !tbaa !24
  %71 = load i32, ptr %r, align 4, !dbg !102, !tbaa !24
  %72 = load i32, ptr %n.addr, align 4, !dbg !102, !tbaa !24
  %xor69 = xor i32 %71, %72, !dbg !102
  %mul70 = mul nsw i32 %xor69, 33, !dbg !109
  store i32 %mul70, ptr %r, align 4, !dbg !110, !tbaa !24
  %73 = load i32, ptr %r, align 4, !dbg !111, !tbaa !24
  %74 = load i32, ptr %n.addr, align 4, !dbg !111, !tbaa !24
  %xor71 = xor i32 %73, %74, !dbg !111
  %mul72 = mul nsw i32 %xor71, 33, !dbg !112
  store i32 %mul72, ptr %r, align 4, !dbg !113, !tbaa !24
  %75 = load i32, ptr %r, align 4, !dbg !111, !tbaa !24
  %76 = load i32, ptr %n.addr, align 4, !dbg !111, !tbaa !24
  %xor73 = xor i32 %75, %76, !dbg !111
  %mul74 = mul nsw i32 %xor73, 33, !dbg !114
  store i32 %mul74, ptr %r, align 4, !dbg !115, !tbaa !24
  %77 = load i32, ptr %r, align 4, !dbg !111, !tbaa !24
  %78 = load i32, ptr %n.addr, align 4, !dbg !111, !tbaa !24
  %xor75 = xor i32 %77, %78, !dbg !111
  %mul76 = mul nsw i32 %xor75, 33, !dbg !116
  store i32 %mul76, ptr %r, align 4, !dbg !117, !tbaa !24
  %79 = load i32, ptr %r, align 4, !dbg !111, !tbaa !24
  %80 = load i32, ptr %n.addr, align 4, !dbg !111, !tbaa !24
  %xor77 = xor i32 %79, %80, !dbg !111
  %mul78 = mul nsw i32 %xor77, 33, !dbg !118
  store i32 %mul78, ptr %r, align 4, !dbg !119, !tbaa !24
  %81 = load i32, ptr %r, align 4, !dbg !120, !tbaa !24
  call void @llvm.lifetime.end.p0(ptr %r) #2, !dbg !121
  ret i32 %81, !dbg !122
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nounwind uwtable
define dso_local noundef i32 @_Z7wrapperi(i32 noundef %n) #0 !dbg !123 {
entry:
  %n.addr = alloca i32, align 4
  store i32 %n, ptr %n.addr, align 4, !tbaa !24
    #dbg_declare(ptr %n.addr, !125, !DIExpression(), !126)
  %0 = load i32, ptr %n.addr, align 4, !dbg !127, !tbaa !24
  %call = call noundef i32 @_Z5heavyIN3lib2v23TagEEii(i32 noundef %0), !dbg !128
  ret i32 %call, !dbg !129
}

attributes #0 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "use-sample-profile" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4}
!llvm.ident = !{!5}
!llvm.errno.tbaa = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 24.0.0git (git@github.com:HighW4y2H3ll/llvm-project.git 671765346dbba0840bb06b6141d5456460619754)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "src/heavy.cpp", directory: "/home/h2h/test/repro_import_guid_tiny", checksumkind: CSK_MD5, checksum: "629dc399bfe92afc397143217a576096")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 7, !"uwtable", i32 2}
!5 = !{!"clang version 24.0.0git (git@github.com:HighW4y2H3ll/llvm-project.git 671765346dbba0840bb06b6141d5456460619754)"}
!6 = !{!7, !8, i64 0}
!7 = !{!"__libc_errno", !8, i64 0}
!8 = !{!"int", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C++ TBAA"}
!11 = distinct !DISubprogram(name: "heavy<lib::v2::Tag>", linkageName: "_Z5heavyIN3lib2v23TagEEii", scope: !1, file: !1, line: 6, type: !12, scopeLine: 6, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, templateParams: !18, retainedNodes: !15, keyInstructions: true)
!12 = !DISubroutineType(types: !13)
!13 = !{!14, !14}
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!15 = !{!16, !17}
!16 = !DILocalVariable(name: "n", arg: 1, scope: !11, file: !1, line: 6, type: !14)
!17 = !DILocalVariable(name: "r", scope: !11, file: !1, line: 6, type: !14)
!18 = !{!19}
!19 = !DITemplateTypeParameter(name: "T", type: !20)
!20 = !DICompositeType(tag: DW_TAG_structure_type, name: "Tag", scope: !22, file: !21, line: 8, size: 8, flags: DIFlagFwdDecl, identifier: "_ZTSN3lib2v23TagE")
!21 = !DIFile(filename: "src/lib.h", directory: "/home/h2h/test/repro_import_guid_tiny", checksumkind: CSK_MD5, checksum: "c696fbf8e2efefb4f774df1237cd7af1")
!22 = !DINamespace(name: "v2", scope: !23, exportSymbols: true)
!23 = !DINamespace(name: "lib", scope: null)
!24 = !{!8, !8, i64 0}
!25 = !DILocation(line: 6, column: 34, scope: !11)
!26 = !DILocation(line: 6, column: 39, scope: !11)
!27 = !DILocation(line: 6, column: 43, scope: !11)
!28 = !DILocation(line: 6, column: 47, scope: !11, atomGroup: 1, atomRank: 2)
!29 = !DILocation(line: 6, column: 43, scope: !11, atomGroup: 1, atomRank: 1)
!30 = !DILocation(line: 6, column: 50, scope: !11)
!31 = !DILocation(line: 6, column: 50, scope: !11, atomGroup: 2, atomRank: 2)
!32 = !DILocation(line: 6, column: 50, scope: !11, atomGroup: 2, atomRank: 1)
!33 = !DILocation(line: 6, column: 50, scope: !11, atomGroup: 3, atomRank: 2)
!34 = !DILocation(line: 6, column: 50, scope: !11, atomGroup: 3, atomRank: 1)
!35 = !DILocation(line: 6, column: 50, scope: !11, atomGroup: 4, atomRank: 2)
!36 = !DILocation(line: 6, column: 50, scope: !11, atomGroup: 4, atomRank: 1)
!37 = !DILocation(line: 6, column: 50, scope: !11, atomGroup: 5, atomRank: 2)
!38 = !DILocation(line: 6, column: 50, scope: !11, atomGroup: 5, atomRank: 1)
!39 = !DILocation(line: 6, column: 53, scope: !11)
!40 = !DILocation(line: 6, column: 53, scope: !11, atomGroup: 6, atomRank: 2)
!41 = !DILocation(line: 6, column: 53, scope: !11, atomGroup: 6, atomRank: 1)
!42 = !DILocation(line: 6, column: 53, scope: !11, atomGroup: 7, atomRank: 2)
!43 = !DILocation(line: 6, column: 53, scope: !11, atomGroup: 7, atomRank: 1)
!44 = !DILocation(line: 6, column: 53, scope: !11, atomGroup: 8, atomRank: 2)
!45 = !DILocation(line: 6, column: 53, scope: !11, atomGroup: 8, atomRank: 1)
!46 = !DILocation(line: 6, column: 53, scope: !11, atomGroup: 9, atomRank: 2)
!47 = !DILocation(line: 6, column: 53, scope: !11, atomGroup: 9, atomRank: 1)
!48 = !DILocation(line: 6, column: 56, scope: !11)
!49 = !DILocation(line: 6, column: 56, scope: !11, atomGroup: 10, atomRank: 2)
!50 = !DILocation(line: 6, column: 56, scope: !11, atomGroup: 10, atomRank: 1)
!51 = !DILocation(line: 6, column: 56, scope: !11, atomGroup: 11, atomRank: 2)
!52 = !DILocation(line: 6, column: 56, scope: !11, atomGroup: 11, atomRank: 1)
!53 = !DILocation(line: 6, column: 56, scope: !11, atomGroup: 12, atomRank: 2)
!54 = !DILocation(line: 6, column: 56, scope: !11, atomGroup: 12, atomRank: 1)
!55 = !DILocation(line: 6, column: 56, scope: !11, atomGroup: 13, atomRank: 2)
!56 = !DILocation(line: 6, column: 56, scope: !11, atomGroup: 13, atomRank: 1)
!57 = !DILocation(line: 6, column: 59, scope: !11)
!58 = !DILocation(line: 6, column: 59, scope: !11, atomGroup: 14, atomRank: 2)
!59 = !DILocation(line: 6, column: 59, scope: !11, atomGroup: 14, atomRank: 1)
!60 = !DILocation(line: 6, column: 59, scope: !11, atomGroup: 15, atomRank: 2)
!61 = !DILocation(line: 6, column: 59, scope: !11, atomGroup: 15, atomRank: 1)
!62 = !DILocation(line: 6, column: 59, scope: !11, atomGroup: 16, atomRank: 2)
!63 = !DILocation(line: 6, column: 59, scope: !11, atomGroup: 16, atomRank: 1)
!64 = !DILocation(line: 6, column: 59, scope: !11, atomGroup: 17, atomRank: 2)
!65 = !DILocation(line: 6, column: 59, scope: !11, atomGroup: 17, atomRank: 1)
!66 = !DILocation(line: 6, column: 62, scope: !11)
!67 = !DILocation(line: 6, column: 62, scope: !11, atomGroup: 18, atomRank: 2)
!68 = !DILocation(line: 6, column: 62, scope: !11, atomGroup: 18, atomRank: 1)
!69 = !DILocation(line: 6, column: 62, scope: !11, atomGroup: 19, atomRank: 2)
!70 = !DILocation(line: 6, column: 62, scope: !11, atomGroup: 19, atomRank: 1)
!71 = !DILocation(line: 6, column: 62, scope: !11, atomGroup: 20, atomRank: 2)
!72 = !DILocation(line: 6, column: 62, scope: !11, atomGroup: 20, atomRank: 1)
!73 = !DILocation(line: 6, column: 62, scope: !11, atomGroup: 21, atomRank: 2)
!74 = !DILocation(line: 6, column: 62, scope: !11, atomGroup: 21, atomRank: 1)
!75 = !DILocation(line: 6, column: 65, scope: !11)
!76 = !DILocation(line: 6, column: 65, scope: !11, atomGroup: 22, atomRank: 2)
!77 = !DILocation(line: 6, column: 65, scope: !11, atomGroup: 22, atomRank: 1)
!78 = !DILocation(line: 6, column: 65, scope: !11, atomGroup: 23, atomRank: 2)
!79 = !DILocation(line: 6, column: 65, scope: !11, atomGroup: 23, atomRank: 1)
!80 = !DILocation(line: 6, column: 65, scope: !11, atomGroup: 24, atomRank: 2)
!81 = !DILocation(line: 6, column: 65, scope: !11, atomGroup: 24, atomRank: 1)
!82 = !DILocation(line: 6, column: 65, scope: !11, atomGroup: 25, atomRank: 2)
!83 = !DILocation(line: 6, column: 65, scope: !11, atomGroup: 25, atomRank: 1)
!84 = !DILocation(line: 6, column: 68, scope: !11)
!85 = !DILocation(line: 6, column: 68, scope: !11, atomGroup: 26, atomRank: 2)
!86 = !DILocation(line: 6, column: 68, scope: !11, atomGroup: 26, atomRank: 1)
!87 = !DILocation(line: 6, column: 68, scope: !11, atomGroup: 27, atomRank: 2)
!88 = !DILocation(line: 6, column: 68, scope: !11, atomGroup: 27, atomRank: 1)
!89 = !DILocation(line: 6, column: 68, scope: !11, atomGroup: 28, atomRank: 2)
!90 = !DILocation(line: 6, column: 68, scope: !11, atomGroup: 28, atomRank: 1)
!91 = !DILocation(line: 6, column: 68, scope: !11, atomGroup: 29, atomRank: 2)
!92 = !DILocation(line: 6, column: 68, scope: !11, atomGroup: 29, atomRank: 1)
!93 = !DILocation(line: 6, column: 71, scope: !11)
!94 = !DILocation(line: 6, column: 71, scope: !11, atomGroup: 30, atomRank: 2)
!95 = !DILocation(line: 6, column: 71, scope: !11, atomGroup: 30, atomRank: 1)
!96 = !DILocation(line: 6, column: 71, scope: !11, atomGroup: 31, atomRank: 2)
!97 = !DILocation(line: 6, column: 71, scope: !11, atomGroup: 31, atomRank: 1)
!98 = !DILocation(line: 6, column: 71, scope: !11, atomGroup: 32, atomRank: 2)
!99 = !DILocation(line: 6, column: 71, scope: !11, atomGroup: 32, atomRank: 1)
!100 = !DILocation(line: 6, column: 71, scope: !11, atomGroup: 33, atomRank: 2)
!101 = !DILocation(line: 6, column: 71, scope: !11, atomGroup: 33, atomRank: 1)
!102 = !DILocation(line: 6, column: 74, scope: !11)
!103 = !DILocation(line: 6, column: 74, scope: !11, atomGroup: 34, atomRank: 2)
!104 = !DILocation(line: 6, column: 74, scope: !11, atomGroup: 34, atomRank: 1)
!105 = !DILocation(line: 6, column: 74, scope: !11, atomGroup: 35, atomRank: 2)
!106 = !DILocation(line: 6, column: 74, scope: !11, atomGroup: 35, atomRank: 1)
!107 = !DILocation(line: 6, column: 74, scope: !11, atomGroup: 36, atomRank: 2)
!108 = !DILocation(line: 6, column: 74, scope: !11, atomGroup: 36, atomRank: 1)
!109 = !DILocation(line: 6, column: 74, scope: !11, atomGroup: 37, atomRank: 2)
!110 = !DILocation(line: 6, column: 74, scope: !11, atomGroup: 37, atomRank: 1)
!111 = !DILocation(line: 6, column: 77, scope: !11)
!112 = !DILocation(line: 6, column: 77, scope: !11, atomGroup: 38, atomRank: 2)
!113 = !DILocation(line: 6, column: 77, scope: !11, atomGroup: 38, atomRank: 1)
!114 = !DILocation(line: 6, column: 77, scope: !11, atomGroup: 39, atomRank: 2)
!115 = !DILocation(line: 6, column: 77, scope: !11, atomGroup: 39, atomRank: 1)
!116 = !DILocation(line: 6, column: 77, scope: !11, atomGroup: 40, atomRank: 2)
!117 = !DILocation(line: 6, column: 77, scope: !11, atomGroup: 40, atomRank: 1)
!118 = !DILocation(line: 6, column: 77, scope: !11, atomGroup: 41, atomRank: 2)
!119 = !DILocation(line: 6, column: 77, scope: !11, atomGroup: 41, atomRank: 1)
!120 = !DILocation(line: 6, column: 87, scope: !11, atomGroup: 43, atomRank: 2)
!121 = !DILocation(line: 6, column: 90, scope: !11)
!122 = !DILocation(line: 6, column: 80, scope: !11, atomGroup: 43, atomRank: 1)
!123 = distinct !DISubprogram(name: "wrapper", linkageName: "_Z7wrapperi", scope: !1, file: !1, line: 11, type: !12, scopeLine: 11, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !124, keyInstructions: true)
!124 = !{!125}
!125 = !DILocalVariable(name: "n", arg: 1, scope: !123, file: !1, line: 11, type: !14)
!126 = !DILocation(line: 11, column: 17, scope: !123)
!127 = !DILocation(line: 11, column: 45, scope: !123)
!128 = !DILocation(line: 11, column: 29, scope: !123, atomGroup: 1, atomRank: 2)
!129 = !DILocation(line: 11, column: 22, scope: !123, atomGroup: 1, atomRank: 1)
