; RUN: llc -O1 -mcpu=gfx1030 -mtriple=amdgcn-amd-amdhsa -filetype=obj -o - < %s | llvm-dwarfdump --debug-info - | FileCheck %s

;; Verify that we produce valid debug locations for parameters of various types.

@glob_ptr = global ptr addrspace(1) null

; CHECK-LABEL: DW_AT_name ("int32_k")
define amdgpu_kernel void @int32_k(i32 %a) !dbg !9 {
  ; CHECK: DW_AT_location
  ; CHECK-NEXT: [0x{{[0-9a-z]+}}, 0x{{[0-9a-z]+}}): DW_OP_regx SGPR{{[0-9]+}})
  tail call void @llvm.dbg.value(metadata i32 %a, metadata !12, metadata !DIExpression(DIOpArg(0, i32))), !dbg !14
  store i32 %a, ptr @glob_ptr, align 4, !dbg !14
  ret void, !dbg !15
}

; CHECK-LABEL: DW_AT_name ("int64_k")
define amdgpu_kernel void @int64_k(i64 %a) !dbg !31 {
  ; CHECK: DW_AT_location
  ; CHECK-NEXT: DW_OP_regx SGPR{{[0-9a-z]+}}, DW_OP_piece 0x4, DW_OP_regx SGPR{{[0-9a-z]+}}, DW_OP_piece 0x4, DW_OP_LLVM_user DW_OP_LLVM_piece_end
  tail call void @llvm.dbg.value(metadata i64 %a, metadata !32, metadata !DIExpression(DIOpArg(0, i64))), !dbg !33
  store i64 %a, ptr @glob_ptr, align 8, !dbg !33
  ret void, !dbg !33
}

; CHECK-LABEL: DW_AT_name ("as1_ptr")
define void @as1_ptr(ptr addrspace(1) %ptr) !dbg !16 {
  ; CHECK: DW_AT_location
  ; CHECK-NEXT: [0x{{[0-9a-z]+}}, 0x{{[0-9a-z]+}}): DW_OP_regx 0x{{[0-9a-z]+}}, DW_OP_LLVM_user DW_OP_LLVM_push_lane, DW_OP_lit4, DW_OP_mul, DW_OP_LLVM_user DW_OP_LLVM_offset, DW_OP_piece 0x4, DW_OP_regx 0x{{[0-9a-z]+}}, DW_OP_LLVM_user DW_OP_LLVM_push_lane, DW_OP_lit4, DW_OP_mul, DW_OP_LLVM_user DW_OP_LLVM_offset, DW_OP_piece 0x4)
  tail call void @llvm.dbg.value(metadata ptr addrspace(1) %ptr, metadata !17, metadata !DIExpression(DIOpArg(0, ptr addrspace(1)))), !dbg !20
  store ptr addrspace(1) %ptr, ptr @glob_ptr, align 8, !dbg !20
  ret void, !dbg !20
}

; CHECK-LABEL: DW_AT_name ("int64")
define void @int64(i64 %a) !dbg !21 {
  ; CHECK: DW_AT_location
  ; CHECK-NEXT: [0x{{[0-9a-z]+}}, 0x{{[0-9a-z]+}}): DW_OP_regx 0x{{[0-9a-z]+}}, DW_OP_LLVM_user DW_OP_LLVM_push_lane, DW_OP_lit4, DW_OP_mul, DW_OP_LLVM_user DW_OP_LLVM_offset, DW_OP_piece 0x4, DW_OP_regx 0x{{[0-9a-z]+}}, DW_OP_LLVM_user DW_OP_LLVM_push_lane, DW_OP_lit4, DW_OP_mul, DW_OP_LLVM_user DW_OP_LLVM_offset, DW_OP_piece 0x4)
  tail call void @llvm.dbg.value(metadata i64 %a, metadata !22, metadata !DIExpression(DIOpArg(0, i64))), !dbg !23
  store i64 %a, ptr @glob_ptr, align 8, !dbg !23
  ret void, !dbg !24
}

; CHECK-LABEL: DW_AT_name ("int32")
define void @int32(i32 %a) !dbg !25 {
  ; CHECK: DW_AT_location (DW_OP_regx 0x{{[0-9a-z]+}}, DW_OP_LLVM_user DW_OP_LLVM_push_lane, DW_OP_lit4, DW_OP_mul, DW_OP_LLVM_user DW_OP_LLVM_offset)
  tail call void @llvm.dbg.value(metadata i32 %a, metadata !26, metadata !DIExpression(DIOpArg(0, i32))), !dbg !27
  store i32 %a, ptr @glob_ptr, align 4, !dbg !27
  ret void, !dbg !27
}

; CHECK-LABEL: DW_AT_name ("gen_ptr")
define void @gen_ptr(ptr %ptr) !dbg !28 {
  ; CHECK: DW_AT_location
  ; CHECK-NEXT: [0x{{[0-9a-z]+}}, 0x{{[0-9a-z]+}}): DW_OP_regx 0x{{[0-9a-z]+}}, DW_OP_LLVM_user DW_OP_LLVM_push_lane, DW_OP_lit4, DW_OP_mul, DW_OP_LLVM_user DW_OP_LLVM_offset, DW_OP_piece 0x4, DW_OP_regx 0x{{[0-9a-z]+}}, DW_OP_LLVM_user DW_OP_LLVM_push_lane, DW_OP_lit4, DW_OP_mul, DW_OP_LLVM_user DW_OP_LLVM_offset, DW_OP_piece 0x4)
  tail call void @llvm.dbg.value(metadata ptr %ptr, metadata !29, metadata !DIExpression(DIOpArg(0, ptr))), !dbg !30
  store ptr %ptr, ptr @glob_ptr, align 8, !dbg !30
  ret void, !dbg !30
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 19.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "t.cpp", directory: "/")
!2 = !{i32 1, !"amdhsa_code_object_version", i32 500}
!3 = !{i32 7, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 8, !"PIC Level", i32 2}
!7 = !{i32 7, !"frame-pointer", i32 2}
!8 = !{!"clang version 19.0.0"}
!9 = distinct !DISubprogram(name: "int32_k", linkageName: "int32_k", scope: !1, file: !1, line: 1, type: !10, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!10 = !DISubroutineType(types: !11)
!11 = !{}
!12 = !DILocalVariable(name: "i32", arg: 1, scope: !9, file: !1, type: !13)
!13 = !DIBasicType(name: "i32", size: 32, encoding: DW_ATE_signed)
!14 = !DILocation(line: 1, column: 1, scope: !9)
!15 = !DILocation(line: 2, column: 1, scope: !9)
!16 = distinct !DISubprogram(name: "as1_ptr", linkageName: "as1_ptr", scope: !1, file: !1, line: 1, type: !10, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!17 = !DILocalVariable(name: "ptr", arg: 1, scope: !16, file: !1, line: 1, type: !18)
!18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !19, size: 64)
!19 = !DIBasicType(name: "i64", size: 64, encoding: DW_ATE_signed)
!20 = !DILocation(line: 1, column: 1, scope: !16)
!21 = distinct !DISubprogram(name: "int64", linkageName: "int64", scope: !1, file: !1, line: 1, type: !10, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!22 = !DILocalVariable(name: "i64", arg: 1, scope: !21, file: !1, type: !19)
!23 = !DILocation(line: 1, column: 1, scope: !21)
!24 = !DILocation(line: 2, column: 1, scope: !21)
!25 = distinct !DISubprogram(name: "int32", linkageName: "int32", scope: !1, file: !1, line: 1, type: !10, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!26 = !DILocalVariable(name: "i32", arg: 1, scope: !25, file: !1, type: !13)
!27 = !DILocation(line: 1, column: 1, scope: !25)
!28 = distinct !DISubprogram(name: "gen_ptr", linkageName: "gen_ptr", scope: !1, file: !1, line: 1, type: !10, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!29 = !DILocalVariable(name: "ptr", arg: 1, scope: !28, file: !1, type: !18)
!30 = !DILocation(line: 1, column: 1, scope: !28)
!31 = distinct !DISubprogram(name: "int64_k", linkageName: "int64_k", scope: !1, file: !1, line: 1, type: !10, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!32 = !DILocalVariable(name: "i32", arg: 1, scope: !31, file: !1, type: !19)
!33 = !DILocation(line: 1, column: 1, scope: !31)
