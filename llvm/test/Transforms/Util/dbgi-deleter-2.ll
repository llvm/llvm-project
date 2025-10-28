; RUN: opt -S -passes=dbg-deleter < %s 2>&1 | FileCheck %s  

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.student = type { [50 x i8], i32, float }

@s = dso_local local_unnamed_addr global [10 x %struct.student] zeroinitializer, align 16, !dbg !0

define dso_local noundef i32 @main() local_unnamed_addr #0 !dbg !28 {
    #dbg_value(i32 2, !32, !DIExpression(), !43)
    #dbg_value(i32 3, !33, !DIExpression(), !43)
    #dbg_value(i32 4, !34, !DIExpression(), !43)
    #dbg_value(ptr poison, !35, !DIExpression(), !43)
    #dbg_value(ptr poison, !37, !DIExpression(), !43)
    #dbg_value(ptr poison, !38, !DIExpression(), !43)
    #dbg_value(i32 5, !34, !DIExpression(), !43)
    #dbg_value(i32 0, !39, !DIExpression(), !44)
    #dbg_value(i64 0, !39, !DIExpression(), !44)
  store i32 6, ptr getelementptr inbounds nuw (i8, ptr @s, i64 52), align 4, !dbg !45, !tbaa !48
    #dbg_value(i64 1, !39, !DIExpression(), !44)
  store i32 7, ptr getelementptr inbounds nuw (i8, ptr @s, i64 112), align 16, !dbg !54, !tbaa !48
    #dbg_value(i64 2, !39, !DIExpression(), !44)
  store i32 8, ptr getelementptr inbounds nuw (i8, ptr @s, i64 172), align 4, !dbg !55, !tbaa !48
    #dbg_value(i64 3, !39, !DIExpression(), !44)
  store i32 9, ptr getelementptr inbounds nuw (i8, ptr @s, i64 232), align 8, !dbg !56, !tbaa !48
    #dbg_value(i64 4, !39, !DIExpression(), !44)
  store i32 10, ptr getelementptr inbounds nuw (i8, ptr @s, i64 292), align 4, !dbg !57, !tbaa !48
    #dbg_value(i64 5, !39, !DIExpression(), !44)
  store i32 11, ptr getelementptr inbounds nuw (i8, ptr @s, i64 352), align 16, !dbg !58, !tbaa !48
    #dbg_value(i64 6, !39, !DIExpression(), !44)
  store i32 12, ptr getelementptr inbounds nuw (i8, ptr @s, i64 412), align 4, !dbg !59, !tbaa !48
    #dbg_value(i64 7, !39, !DIExpression(), !44)
  store i32 13, ptr getelementptr inbounds nuw (i8, ptr @s, i64 472), align 8, !dbg !60, !tbaa !48
    #dbg_value(i64 8, !39, !DIExpression(), !44)
  store i32 14, ptr getelementptr inbounds nuw (i8, ptr @s, i64 532), align 4, !dbg !61, !tbaa !48
    #dbg_value(i64 9, !39, !DIExpression(), !44)
  store i32 15, ptr getelementptr inbounds nuw (i8, ptr @s, i64 592), align 16, !dbg !62, !tbaa !48
    #dbg_value(i64 10, !39, !DIExpression(), !44)
    #dbg_value(i64 0, !41, !DIExpression(), !63)
  %1 = load float, ptr getelementptr inbounds nuw (i8, ptr @s, i64 56), align 8, !dbg !64, !tbaa !67
  %2 = fadd float %1, 1.000000e+00, !dbg !68
  store float %2, ptr getelementptr inbounds nuw (i8, ptr @s, i64 56), align 8, !dbg !69, !tbaa !67
    #dbg_value(i64 1, !41, !DIExpression(), !63)
  %3 = load float, ptr getelementptr inbounds nuw (i8, ptr @s, i64 116), align 4, !dbg !64, !tbaa !67
  %4 = fadd float %3, 1.000000e+00, !dbg !70
  store float %4, ptr getelementptr inbounds nuw (i8, ptr @s, i64 116), align 4, !dbg !71, !tbaa !67
    #dbg_value(i64 2, !41, !DIExpression(), !63)
  %5 = load float, ptr getelementptr inbounds nuw (i8, ptr @s, i64 176), align 16, !dbg !64, !tbaa !67
  %6 = fadd float %5, 1.000000e+00, !dbg !72
  store float %6, ptr getelementptr inbounds nuw (i8, ptr @s, i64 176), align 16, !dbg !73, !tbaa !67
    #dbg_value(i64 3, !41, !DIExpression(), !63)
  %7 = load float, ptr getelementptr inbounds nuw (i8, ptr @s, i64 236), align 4, !dbg !64, !tbaa !67
  %8 = fadd float %7, 1.000000e+00, !dbg !74
  store float %8, ptr getelementptr inbounds nuw (i8, ptr @s, i64 236), align 4, !dbg !75, !tbaa !67
    #dbg_value(i64 4, !41, !DIExpression(), !63)
  %9 = load float, ptr getelementptr inbounds nuw (i8, ptr @s, i64 296), align 8, !dbg !64, !tbaa !67
  %10 = fadd float %9, 1.000000e+00, !dbg !76
  store float %10, ptr getelementptr inbounds nuw (i8, ptr @s, i64 296), align 8, !dbg !77, !tbaa !67
    #dbg_value(i64 5, !41, !DIExpression(), !63)
  %11 = load float, ptr getelementptr inbounds nuw (i8, ptr @s, i64 356), align 4, !dbg !64, !tbaa !67
  %12 = fadd float %11, 1.000000e+00, !dbg !78
  store float %12, ptr getelementptr inbounds nuw (i8, ptr @s, i64 356), align 4, !dbg !79, !tbaa !67
    #dbg_value(i64 6, !41, !DIExpression(), !63)
  %13 = load float, ptr getelementptr inbounds nuw (i8, ptr @s, i64 416), align 16, !dbg !64, !tbaa !67
  %14 = fadd float %13, 1.000000e+00, !dbg !80
  store float %14, ptr getelementptr inbounds nuw (i8, ptr @s, i64 416), align 16, !dbg !81, !tbaa !67
    #dbg_value(i64 7, !41, !DIExpression(), !63)
  %15 = load float, ptr getelementptr inbounds nuw (i8, ptr @s, i64 476), align 4, !dbg !64, !tbaa !67
  %16 = fadd float %15, 1.000000e+00, !dbg !82
  store float %16, ptr getelementptr inbounds nuw (i8, ptr @s, i64 476), align 4, !dbg !83, !tbaa !67
    #dbg_value(i64 8, !41, !DIExpression(), !63)
  %17 = load float, ptr getelementptr inbounds nuw (i8, ptr @s, i64 536), align 8, !dbg !64, !tbaa !67
  %18 = fadd float %17, 1.000000e+00, !dbg !84
  store float %18, ptr getelementptr inbounds nuw (i8, ptr @s, i64 536), align 8, !dbg !85, !tbaa !67
    #dbg_value(i64 9, !41, !DIExpression(), !63)
  %19 = load float, ptr getelementptr inbounds nuw (i8, ptr @s, i64 596), align 4, !dbg !64, !tbaa !67
  %20 = fadd float %19, 1.000000e+00, !dbg !86
  store float %20, ptr getelementptr inbounds nuw (i8, ptr @s, i64 596), align 4, !dbg !87, !tbaa !67
    #dbg_value(i64 10, !41, !DIExpression(), !63)
  ret i32 0, !dbg !88

; CHECK: @main()
; CHECK-NOT: #dbg_value
; CHECK-NOT: !dbg
; CHECK: ret i32

}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!20, !21, !22, !23, !24, !25, !26}
!llvm.ident = !{!27}

; CHECK: attributes #0
; CHECK-NOT: !llvm.dbg.cu
; CHECK: !llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6}
; CHECK: !llvm.ident = !{!7}


!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "s", scope: !2, file: !5, line: 8, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 22.0.0git (https://github.com/anamaoh/llvm-project.git 782a91e1fc94d9c82495f60afc5ed5edd72de776)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "/home/ana-marija/Documents/load.cpp", directory: "/home/ana-marija/Desktop/LLVM/llvm-project", checksumkind: CSK_MD5, checksum: "1bce1e274606359bd3ee799e44db5ab7")
!4 = !{!0}
!5 = !DIFile(filename: "Documents/load.cpp", directory: "/home/ana-marija", checksumkind: CSK_MD5, checksum: "1bce1e274606359bd3ee799e44db5ab7")
!6 = !DICompositeType(tag: DW_TAG_array_type, baseType: !7, size: 4800, elements: !18)
!7 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "student", file: !5, line: 3, size: 480, flags: DIFlagTypePassByValue, elements: !8, identifier: "_ZTS7student")
!8 = !{!9, !14, !16}
!9 = !DIDerivedType(tag: DW_TAG_member, name: "name", scope: !7, file: !5, line: 5, baseType: !10, size: 400)
!10 = !DICompositeType(tag: DW_TAG_array_type, baseType: !11, size: 400, elements: !12)
!11 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!12 = !{!13}
!13 = !DISubrange(count: 50)
!14 = !DIDerivedType(tag: DW_TAG_member, name: "roll", scope: !7, file: !5, line: 6, baseType: !15, size: 32, offset: 416)
!15 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!16 = !DIDerivedType(tag: DW_TAG_member, name: "marks", scope: !7, file: !5, line: 7, baseType: !17, size: 32, offset: 448)
!17 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!18 = !{!19}
!19 = !DISubrange(count: 10)
!20 = !{i32 7, !"Dwarf Version", i32 5}
!21 = !{i32 2, !"Debug Info Version", i32 3}
!22 = !{i32 1, !"wchar_size", i32 4}
!23 = !{i32 8, !"PIC Level", i32 2}
!24 = !{i32 7, !"PIE Level", i32 2}
!25 = !{i32 7, !"uwtable", i32 2}
!26 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!27 = !{!"clang version 22.0.0git (https://github.com/anamaoh/llvm-project.git 782a91e1fc94d9c82495f60afc5ed5edd72de776)"}
!28 = distinct !DISubprogram(name: "main", scope: !5, file: !5, line: 10, type: !29, scopeLine: 11, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !31, keyInstructions: true)
!29 = !DISubroutineType(types: !30)
!30 = !{!15}
!31 = !{!32, !33, !34, !35, !37, !38, !39, !41}
!32 = !DILocalVariable(name: "x", scope: !28, file: !5, line: 12, type: !15)
!33 = !DILocalVariable(name: "y", scope: !28, file: !5, line: 12, type: !15)
!34 = !DILocalVariable(name: "z", scope: !28, file: !5, line: 12, type: !15)
!35 = !DILocalVariable(name: "px", scope: !28, file: !5, line: 13, type: !36)
!36 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !15, size: 64)
!37 = !DILocalVariable(name: "py", scope: !28, file: !5, line: 13, type: !36)
!38 = !DILocalVariable(name: "pz", scope: !28, file: !5, line: 13, type: !36)
!39 = !DILocalVariable(name: "i", scope: !40, file: !5, line: 17, type: !15)
!40 = distinct !DILexicalBlock(scope: !28, file: !5, line: 17, column: 5)
!41 = !DILocalVariable(name: "i", scope: !42, file: !5, line: 23, type: !15)
!42 = distinct !DILexicalBlock(scope: !28, file: !5, line: 23, column: 5)
!43 = !DILocation(line: 0, scope: !28)
!44 = !DILocation(line: 0, scope: !40)
!45 = !DILocation(line: 19, column: 19, scope: !46, atomGroup: 11, atomRank: 1)
!46 = distinct !DILexicalBlock(scope: !47, file: !5, line: 18, column: 5)
!47 = distinct !DILexicalBlock(scope: !40, file: !5, line: 17, column: 5)
!48 = !{!49, !52, i64 52}
!49 = !{!"_ZTS7student", !50, i64 0, !52, i64 52, !53, i64 56}
!50 = !{!"omnipotent char", !51, i64 0}
!51 = !{!"Simple C++ TBAA"}
!52 = !{!"int", !50, i64 0}
!53 = !{!"float", !50, i64 0}
!54 = !DILocation(line: 19, column: 19, scope: !46, atomGroup: 26, atomRank: 1)
!55 = !DILocation(line: 19, column: 19, scope: !46, atomGroup: 29, atomRank: 1)
!56 = !DILocation(line: 19, column: 19, scope: !46, atomGroup: 32, atomRank: 1)
!57 = !DILocation(line: 19, column: 19, scope: !46, atomGroup: 35, atomRank: 1)
!58 = !DILocation(line: 19, column: 19, scope: !46, atomGroup: 38, atomRank: 1)
!59 = !DILocation(line: 19, column: 19, scope: !46, atomGroup: 41, atomRank: 1)
!60 = !DILocation(line: 19, column: 19, scope: !46, atomGroup: 44, atomRank: 1)
!61 = !DILocation(line: 19, column: 19, scope: !46, atomGroup: 47, atomRank: 1)
!62 = !DILocation(line: 19, column: 19, scope: !46, atomGroup: 50, atomRank: 1)
!63 = !DILocation(line: 0, scope: !42)
!64 = !DILocation(line: 25, column: 19, scope: !65)
!65 = distinct !DILexicalBlock(scope: !66, file: !5, line: 24, column: 5)
!66 = distinct !DILexicalBlock(scope: !42, file: !5, line: 23, column: 5)
!67 = !{!49, !53, i64 56}
!68 = !DILocation(line: 25, column: 19, scope: !65, atomGroup: 17, atomRank: 2)
!69 = !DILocation(line: 25, column: 19, scope: !65, atomGroup: 17, atomRank: 1)
!70 = !DILocation(line: 25, column: 19, scope: !65, atomGroup: 53, atomRank: 2)
!71 = !DILocation(line: 25, column: 19, scope: !65, atomGroup: 53, atomRank: 1)
!72 = !DILocation(line: 25, column: 19, scope: !65, atomGroup: 57, atomRank: 2)
!73 = !DILocation(line: 25, column: 19, scope: !65, atomGroup: 57, atomRank: 1)
!74 = !DILocation(line: 25, column: 19, scope: !65, atomGroup: 61, atomRank: 2)
!75 = !DILocation(line: 25, column: 19, scope: !65, atomGroup: 61, atomRank: 1)
!76 = !DILocation(line: 25, column: 19, scope: !65, atomGroup: 65, atomRank: 2)
!77 = !DILocation(line: 25, column: 19, scope: !65, atomGroup: 65, atomRank: 1)
!78 = !DILocation(line: 25, column: 19, scope: !65, atomGroup: 69, atomRank: 2)
!79 = !DILocation(line: 25, column: 19, scope: !65, atomGroup: 69, atomRank: 1)
!80 = !DILocation(line: 25, column: 19, scope: !65, atomGroup: 73, atomRank: 2)
!81 = !DILocation(line: 25, column: 19, scope: !65, atomGroup: 73, atomRank: 1)
!82 = !DILocation(line: 25, column: 19, scope: !65, atomGroup: 77, atomRank: 2)
!83 = !DILocation(line: 25, column: 19, scope: !65, atomGroup: 77, atomRank: 1)
!84 = !DILocation(line: 25, column: 19, scope: !65, atomGroup: 81, atomRank: 2)
!85 = !DILocation(line: 25, column: 19, scope: !65, atomGroup: 81, atomRank: 1)
!86 = !DILocation(line: 25, column: 19, scope: !65, atomGroup: 85, atomRank: 2)
!87 = !DILocation(line: 25, column: 19, scope: !65, atomGroup: 85, atomRank: 1)
!88 = !DILocation(line: 28, column: 5, scope: !28, atomGroup: 21, atomRank: 1)


; CHECK-NOT: !DICompileUnit
; CHECK-NOT: !DIFile 
; CHECK-NOT: !DISubprogram
; CHECK-NOT: !DIFile 
; CHECK-NOT: !DISubroutineType 
; CHECK-NOT: !DIBasicType
; CHECK-NOT: !DILocalVariable
; CHECK-NOT: !DILocation
; CHECK-NOT: !DIGlobalVariableExpression
; CHECK-NOT: !DIGlobalVariable
; CHECK-NOT: !DICompositeType
; CHECK-NOT: !DIDerivedType
; CHECK-NOT: !DISubrange 
; CHECK-NOT: !DILexicalBlock