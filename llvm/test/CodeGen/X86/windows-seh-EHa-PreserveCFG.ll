; RUN: llc -mtriple=x86_64-pc-windows-msvc %s -o /dev/null
define dso_local void @main(ptr %addr, ptr %src, ptr %dst) personality ptr @__CxxFrameHandler3 !dbg !11 {
entry:
  %tmp0 = load float, ptr %src
  %src1 = getelementptr inbounds float, ptr %src, i64 1
  %tmp1 = load float, ptr %src1
  %src2 = getelementptr inbounds float, ptr %src, i64 2
  %tmp2 = load float, ptr %src2
  %src3 = getelementptr inbounds float, ptr %src, i64 3
  %tmp3 = load float, ptr %src3
  %src4 = getelementptr inbounds float, ptr %src, i64 4
  %tmp4 = load float, ptr %src4
  %src5 = getelementptr inbounds float, ptr %src, i64 5
  %tmp5 = load float, ptr %src5
  %src6 = getelementptr inbounds float, ptr %src, i64 6
  %tmp6 = load float, ptr %src6
  invoke void @foo(ptr %addr)
          to label %scope_begin unwind label %ehcleanup1, !dbg !13

scope_begin:
  invoke void @llvm.seh.scope.begin()
          to label %scope_end unwind label %ehcleanup, !dbg !13

scope_end:
  invoke void @llvm.seh.scope.end()
          to label %finish unwind label %ehcleanup, !dbg !13

ehcleanup:
  %0 = cleanuppad within none [], !dbg !13
  call void @llvm.dbg.value(metadata ptr %addr, metadata !12, metadata !DIExpression()), !dbg !13
  call void @foo(ptr %addr) [ "funclet"(token %0) ], !dbg !13
  cleanupret from %0 unwind label %ehcleanup1, !dbg !13

ehcleanup1:
  %1 = cleanuppad within none [], !dbg !13
  call void @foo(ptr %addr) [ "funclet"(token %1) ], !dbg !13
  cleanupret from %1 unwind to caller, !dbg !13

finish:
  store float %tmp0, ptr %dst
  %dst1 = getelementptr inbounds float, ptr %dst, i64 1
  store float %tmp1, ptr %dst1
  %dst2 = getelementptr inbounds float, ptr %dst, i64 2
  store float %tmp2, ptr %dst2
  %dst3 = getelementptr inbounds float, ptr %dst, i64 3
  store float %tmp3, ptr %dst3
  %dst4 = getelementptr inbounds float, ptr %dst, i64 4
  store float %tmp4, ptr %dst4
  %dst5 = getelementptr inbounds float, ptr %dst, i64 5
  store float %tmp5, ptr %dst5
  %dst6 = getelementptr inbounds float, ptr %dst, i64 6
  store float %tmp6, ptr %dst6
  ret void
}

declare dso_local void @llvm.seh.scope.begin()
declare dso_local void @llvm.seh.scope.end()
declare dso_local i32 @__CxxFrameHandler3(...)
declare dso_local void @foo(ptr %addr)
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.dbg.cu = !{!14}

!0 = !{i32 2, !"eh-asynch", i32 1}
!1 = !{i32 2, !"CodeView", i32 1}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 7, !"uwtable", i32 2}

!4 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!5 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !4, size: 64)
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 64)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !7, !5, !5}
!10 = !DIFile(filename: "c:/main.cpp", directory: "")
!11 = distinct !DISubprogram(name: "main", scope: !10, file: !10, line: 5, type: !8, scopeLine: 11, unit: !14)
!12 = !DILocalVariable(name: "addr", scope: !11, file: !10, line: 5, type: !7)
!13 = !DILocation(line: 7, scope: !11)
!14 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !10, isOptimized: true, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
