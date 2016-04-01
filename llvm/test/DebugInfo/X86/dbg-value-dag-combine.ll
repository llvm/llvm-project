; RUN: llc < %s | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:32"
target triple = "i686-apple-darwin"
; PR 9817


declare <4 x i32> @__amdil_get_global_id_int()
declare void @llvm.dbg.value(metadata, i64, metadata, metadata)
define void @__OpenCL_test_kernel(i32 addrspace(1)* %ip) nounwind !dbg !0 {
entry:
  call void @llvm.dbg.value(metadata i32 addrspace(1)* %ip, i64 0, metadata !7, metadata !DIExpression()), !dbg !8
  %0 = call <4 x i32> @__amdil_get_global_id_int() nounwind
  %1 = extractelement <4 x i32> %0, i32 0
  call void @llvm.dbg.value(metadata i32 %1, i64 0, metadata !9, metadata !DIExpression()), !dbg !11
  call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !13, metadata !DIExpression()), !dbg !14
  %tmp2 = load i32, i32 addrspace(1)* %ip, align 4, !dbg !15
  %tmp3 = add i32 0, %tmp2, !dbg !15
; CHECK:  ##DEBUG_VALUE: idx <- %E{{..$}}
  call void @llvm.dbg.value(metadata i32 %tmp3, i64 0, metadata !13, metadata !DIExpression()), !dbg !15
  %arrayidx = getelementptr i32, i32 addrspace(1)* %ip, i32 %1, !dbg !16
  store i32 %tmp3, i32 addrspace(1)* %arrayidx, align 4, !dbg !16
  ret void, !dbg !17
}
!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!20}

!0 = distinct !DISubprogram(name: "__OpenCL_test_kernel", linkageName: "__OpenCL_test_kernel", line: 2, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, file: !19, scope: !1, type: !3)
!1 = !DIFile(filename: "OCL6368.tmp.cl", directory: "E:CUsersCmvillmow.AMDCAppDataCLocalCTemp")
!2 = distinct !DICompileUnit(language: DW_LANG_C89, producer: "clc", isOptimized: false, emissionKind: FullDebug, file: !19, enums: !12, retainedTypes: !12, subprograms: !18, imports:  null)
!3 = !DISubroutineType(types: !4)
!4 = !{null, !5}
!5 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 32, align: 32, scope: !2, baseType: !6)
!6 = !DIBasicType(tag: DW_TAG_base_type, name: "unsigned int", size: 32, align: 32, encoding: DW_ATE_unsigned)
!7 = !DILocalVariable(name: "ip", line: 1, arg: 1, scope: !0, file: !1, type: !5)
!8 = !DILocation(line: 1, column: 42, scope: !0)
!9 = !DILocalVariable(name: "gid", line: 3, scope: !10, file: !1, type: !6)
!10 = distinct !DILexicalBlock(line: 2, column: 1, file: !19, scope: !0)
!11 = !DILocation(line: 3, column: 41, scope: !10)
!12 = !{}
!13 = !DILocalVariable(name: "idx", line: 4, scope: !10, file: !1, type: !6)
!14 = !DILocation(line: 4, column: 20, scope: !10)
!15 = !DILocation(line: 5, column: 15, scope: !10)
!16 = !DILocation(line: 6, column: 18, scope: !10)
!17 = !DILocation(line: 7, column: 1, scope: !0)
!18 = !{!0}
!19 = !DIFile(filename: "OCL6368.tmp.cl", directory: "E:\5CUsers\5Cmvillmow.AMD\5CAppData\5CLocal\5CTemp")
!20 = !{i32 1, !"Debug Info Version", i32 3}
