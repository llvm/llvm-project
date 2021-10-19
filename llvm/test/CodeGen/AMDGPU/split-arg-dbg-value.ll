; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 < %s | FileCheck -check-prefix=GCN %s
; Make sure dbg_value reports something for argument registers when they are split into multiple registers

define hidden <4 x float> @split_v4f32_arg(<4 x float> returned %arg) local_unnamed_addr #0 !dbg !7 {
; GCN-LABEL: split_v4f32_arg:
; GCN:       .Lfunc_begin0:
; GCN-NEXT:    .file 0
; GCN-NEXT:    .loc 0 3 0 ; /tmp/dbg.cl:3:0
; GCN-NEXT:    .cfi_sections .debug_frame
; GCN-NEXT:    .cfi_startproc
; GCN-NEXT:  ; %bb.0:
; GCN-NEXT:    ;DEBUG_VALUE: split_v4f32_arg:arg <- [DW_OP_constu 1, DW_OP_swap, DW_OP_xderef, DW_OP_LLVM_fragment 96 32] $vgpr3
; GCN-NEXT:    ;DEBUG_VALUE: split_v4f32_arg:arg <- [DW_OP_constu 1, DW_OP_swap, DW_OP_xderef, DW_OP_LLVM_fragment 64 32] $vgpr2
; GCN-NEXT:    ;DEBUG_VALUE: split_v4f32_arg:arg <- [DW_OP_constu 1, DW_OP_swap, DW_OP_xderef, DW_OP_LLVM_fragment 32 32] $vgpr1
; GCN-NEXT:    ;DEBUG_VALUE: split_v4f32_arg:arg <- [DW_OP_constu 1, DW_OP_swap, DW_OP_xderef, DW_OP_LLVM_fragment 0 32] $vgpr0
; GCN-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NEXT:  .Ltmp0:
; GCN-NEXT:    .loc 0 4 5 prologue_end ; /tmp/dbg.cl:4:5
; GCN-NEXT:    s_setpc_b64 s[30:31]
; GCN-NEXT:  .Ltmp1:
; GCN:         .cfi_endproc
  call void @llvm.dbg.value(metadata <4 x float> %arg, metadata !18, metadata !DIExpression(DW_OP_constu, 1, DW_OP_swap, DW_OP_xderef)), !dbg !19
  ret <4 x float> %arg, !dbg !20
}

define hidden <4 x float> @split_v4f32_multi_arg(<4 x float> %arg0, <2 x float> %arg1) local_unnamed_addr #0 !dbg !21 {
; GCN-LABEL: split_v4f32_multi_arg:
; GCN:       .Lfunc_begin1:
; GCN-NEXT:    .loc 0 7 0 ; /tmp/dbg.cl:7:0
; GCN-NEXT:    .cfi_startproc
; GCN-NEXT:  ; %bb.0:
; GCN-NEXT:    ;DEBUG_VALUE: split_v4f32_multi_arg:arg1 <- [DW_OP_constu 1, DW_OP_swap, DW_OP_xderef, DW_OP_LLVM_fragment 32 32] $vgpr5
; GCN-NEXT:    ;DEBUG_VALUE: split_v4f32_multi_arg:arg1 <- [DW_OP_constu 1, DW_OP_swap, DW_OP_xderef, DW_OP_LLVM_fragment 0 32] $vgpr4
; GCN-NEXT:    ;DEBUG_VALUE: split_v4f32_multi_arg:arg0 <- [DW_OP_constu 1, DW_OP_swap, DW_OP_xderef, DW_OP_LLVM_fragment 96 32] $vgpr3
; GCN-NEXT:    ;DEBUG_VALUE: split_v4f32_multi_arg:arg0 <- [DW_OP_constu 1, DW_OP_swap, DW_OP_xderef, DW_OP_LLVM_fragment 64 32] $vgpr2
; GCN-NEXT:    ;DEBUG_VALUE: split_v4f32_multi_arg:arg0 <- [DW_OP_constu 1, DW_OP_swap, DW_OP_xderef, DW_OP_LLVM_fragment 32 32] $vgpr1
; GCN-NEXT:    ;DEBUG_VALUE: split_v4f32_multi_arg:arg0 <- [DW_OP_constu 1, DW_OP_swap, DW_OP_xderef, DW_OP_LLVM_fragment 0 32] $vgpr0
; GCN-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NEXT:  .Ltmp2:
; GCN-NEXT:    .loc 0 8 17 prologue_end ; /tmp/dbg.cl:8:17
; GCN-NEXT:    v_add_f32_e32 v0, v4, v0
; GCN-NEXT:  .Ltmp3:
; GCN-NEXT:    v_add_f32_e32 v1, v5, v1
; GCN-NEXT:  .Ltmp4:
; GCN-NEXT:    v_add_f32_e32 v2, v4, v2
; GCN-NEXT:  .Ltmp5:
; GCN-NEXT:    v_add_f32_e32 v3, v5, v3
; GCN-NEXT:  .Ltmp6:
; GCN-NEXT:    .loc 0 8 5 is_stmt 0 ; /tmp/dbg.cl:8:5
; GCN-NEXT:    s_setpc_b64 s[30:31]
; GCN-NEXT:  .Ltmp7:
; GCN:         .cfi_endproc
  call void @llvm.dbg.value(metadata <4 x float> %arg0, metadata !29, metadata !DIExpression(DW_OP_constu, 1, DW_OP_swap, DW_OP_xderef)), !dbg !31
  call void @llvm.dbg.value(metadata <2 x float> %arg1, metadata !30, metadata !DIExpression(DW_OP_constu, 1, DW_OP_swap, DW_OP_xderef)), !dbg !31
  %tmp = shufflevector <2 x float> %arg1, <2 x float> undef, <4 x i32> <i32 0, i32 1, i32 0, i32 1>, !dbg !32
  %add = fadd <4 x float> %tmp, %arg0, !dbg !33
  ret <4 x float> %add, !dbg !34
}

define hidden <4 x half> @split_v4f16_arg(<4 x half> returned %arg) local_unnamed_addr #0 !dbg !35 {
; GCN-LABEL: split_v4f16_arg:
; GCN:       .Lfunc_begin2:
; GCN-NEXT:    .loc 0 11 0 is_stmt 1 ; /tmp/dbg.cl:11:0
; GCN-NEXT:    .cfi_startproc
; GCN-NEXT:  ; %bb.0:
; GCN-NEXT:    ;DEBUG_VALUE: split_v4f16_arg:arg <- [DW_OP_constu 1, DW_OP_swap, DW_OP_xderef, DW_OP_LLVM_fragment 32 32] $vgpr1
; GCN-NEXT:    ;DEBUG_VALUE: split_v4f16_arg:arg <- [DW_OP_constu 1, DW_OP_swap, DW_OP_xderef, DW_OP_LLVM_fragment 0 32] $vgpr0
; GCN-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NEXT:  .Ltmp8:
; GCN-NEXT:    .loc 0 12 5 prologue_end ; /tmp/dbg.cl:12:5
; GCN-NEXT:    s_setpc_b64 s[30:31]
; GCN-NEXT:  .Ltmp9:
; GCN:         .cfi_endproc
  call void @llvm.dbg.value(metadata <4 x half> %arg, metadata !42, metadata !DIExpression(DW_OP_constu, 1, DW_OP_swap, DW_OP_xderef)), !dbg !43
  ret <4 x half> %arg, !dbg !44
}

define hidden double @split_f64_arg(double returned %arg) local_unnamed_addr #0 !dbg !45 {
; GCN-LABEL: split_f64_arg:
; GCN:       .Lfunc_begin3:
; GCN-NEXT:    .loc 0 15 0 ; /tmp/dbg.cl:15:0
; GCN-NEXT:    .cfi_startproc
; GCN-NEXT:  ; %bb.0:
; GCN-NEXT:    ;DEBUG_VALUE: split_f64_arg:arg <- [DW_OP_constu 1, DW_OP_swap, DW_OP_xderef, DW_OP_LLVM_fragment 32 32] $vgpr1
; GCN-NEXT:    ;DEBUG_VALUE: split_f64_arg:arg <- [DW_OP_constu 1, DW_OP_swap, DW_OP_xderef, DW_OP_LLVM_fragment 0 32] $vgpr0
; GCN-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NEXT:  .Ltmp10:
; GCN-NEXT:    .loc 0 16 5 prologue_end ; /tmp/dbg.cl:16:5
; GCN-NEXT:    s_setpc_b64 s[30:31]
; GCN-NEXT:  .Ltmp11:
; GCN:         .cfi_endproc
  call void @llvm.dbg.value(metadata double %arg, metadata !50, metadata !DIExpression(DW_OP_constu, 1, DW_OP_swap, DW_OP_xderef)), !dbg !51
  ret double %arg, !dbg !52
}

define hidden <2 x double> @split_v2f64_arg(<2 x double> returned %arg) local_unnamed_addr #0 !dbg !53 {
; GCN-LABEL: split_v2f64_arg:
; GCN:       .Lfunc_begin4:
; GCN-NEXT:    .loc 0 19 0 ; /tmp/dbg.cl:19:0
; GCN-NEXT:    .cfi_startproc
; GCN-NEXT:  ; %bb.0:
; GCN-NEXT:    ;DEBUG_VALUE: split_v2f64_arg:arg <- [DW_OP_constu 1, DW_OP_swap, DW_OP_xderef, DW_OP_LLVM_fragment 96 32] $vgpr3
; GCN-NEXT:    ;DEBUG_VALUE: split_v2f64_arg:arg <- [DW_OP_constu 1, DW_OP_swap, DW_OP_xderef, DW_OP_LLVM_fragment 64 32] $vgpr2
; GCN-NEXT:    ;DEBUG_VALUE: split_v2f64_arg:arg <- [DW_OP_constu 1, DW_OP_swap, DW_OP_xderef, DW_OP_LLVM_fragment 32 32] $vgpr1
; GCN-NEXT:    ;DEBUG_VALUE: split_v2f64_arg:arg <- [DW_OP_constu 1, DW_OP_swap, DW_OP_xderef, DW_OP_LLVM_fragment 0 32] $vgpr0
; GCN-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NEXT:  .Ltmp12:
; GCN-NEXT:    .loc 0 20 5 prologue_end ; /tmp/dbg.cl:20:5
; GCN-NEXT:    s_setpc_b64 s[30:31]
; GCN-NEXT:  .Ltmp13:
; GCN:         .cfi_endproc
  call void @llvm.dbg.value(metadata <2 x double> %arg, metadata !59, metadata !DIExpression(DW_OP_constu, 1, DW_OP_swap, DW_OP_xderef)), !dbg !60
  ret <2 x double> %arg, !dbg !61
}

define hidden i64 @split_i64_arg(i64 returned %arg) local_unnamed_addr #0 !dbg !62 {
; GCN-LABEL: split_i64_arg:
; GCN:       .Lfunc_begin5:
; GCN-NEXT:    .loc 0 23 0 ; /tmp/dbg.cl:23:0
; GCN-NEXT:    .cfi_startproc
; GCN-NEXT:  ; %bb.0:
; GCN-NEXT:    ;DEBUG_VALUE: split_i64_arg:arg <- [DW_OP_constu 1, DW_OP_swap, DW_OP_xderef, DW_OP_LLVM_fragment 32 32] $vgpr1
; GCN-NEXT:    ;DEBUG_VALUE: split_i64_arg:arg <- [DW_OP_constu 1, DW_OP_swap, DW_OP_xderef, DW_OP_LLVM_fragment 0 32] $vgpr0
; GCN-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NEXT:  .Ltmp14:
; GCN-NEXT:    .loc 0 24 5 prologue_end ; /tmp/dbg.cl:24:5
; GCN-NEXT:    s_setpc_b64 s[30:31]
; GCN-NEXT:  .Ltmp15:
; GCN:         .cfi_endproc
  call void @llvm.dbg.value(metadata i64 %arg, metadata !67, metadata !DIExpression(DW_OP_constu, 1, DW_OP_swap, DW_OP_xderef)), !dbg !68
  ret i64 %arg, !dbg !69
}

define hidden i8 addrspace(1)* @split_ptr_arg(i8 addrspace(1)* readnone returned %arg) local_unnamed_addr #0 !dbg !70 {
; GCN-LABEL: split_ptr_arg:
; GCN:       .Lfunc_begin6:
; GCN-NEXT:    .loc 0 27 0 ; /tmp/dbg.cl:27:0
; GCN-NEXT:    .cfi_startproc
; GCN-NEXT:  ; %bb.0:
; GCN-NEXT:    ;DEBUG_VALUE: split_ptr_arg:arg <- [DW_OP_constu 1, DW_OP_swap, DW_OP_xderef, DW_OP_LLVM_fragment 32 32] $vgpr1
; GCN-NEXT:    ;DEBUG_VALUE: split_ptr_arg:arg <- [DW_OP_constu 1, DW_OP_swap, DW_OP_xderef, DW_OP_LLVM_fragment 0 32] $vgpr0
; GCN-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NEXT:  .Ltmp16:
; GCN-NEXT:    .loc 0 28 5 prologue_end ; /tmp/dbg.cl:28:5
; GCN-NEXT:    s_setpc_b64 s[30:31]
; GCN-NEXT:  .Ltmp17:
; GCN:         .cfi_endproc
  call void @llvm.dbg.value(metadata i8 addrspace(1)* %arg, metadata !76, metadata !DIExpression(DW_OP_constu, 1, DW_OP_swap, DW_OP_xderef)), !dbg !77
  ret i8 addrspace(1)* %arg, !dbg !78
}

declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { norecurse nounwind readnone }
attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 9.0.0 (trunk 365209) (llvm/trunk 365206)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "/tmp/dbg.cl", directory: "/Users/matt/src/llvm", checksumkind: CSK_MD5, checksum: "0f834f91e91489a5ff6308040ddbd175")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 7, !"PIC Level", i32 1}
!7 = distinct !DISubprogram(name: "split_v4f32_arg", scope: !8, file: !8, line: 3, type: !9, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !17)
!8 = !DIFile(filename: "/tmp/dbg.cl", directory: "", checksumkind: CSK_MD5, checksum: "0f834f91e91489a5ff6308040ddbd175")
!9 = !DISubroutineType(types: !10)
!10 = !{!11, !11}
!11 = !DIDerivedType(tag: DW_TAG_typedef, name: "float4", file: !12, line: 107, baseType: !13)
!12 = !DIFile(filename: "build_debug/lib/clang/9.0.0/include/opencl-c-base.h", directory: "/Users/matt/src/llvm", checksumkind: CSK_MD5, checksum: "9526a66ac52220225f05e11186d7e461")
!13 = !DICompositeType(tag: DW_TAG_array_type, baseType: !14, size: 128, flags: DIFlagVector, elements: !15)
!14 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!15 = !{!16}
!16 = !DISubrange(count: 4)
!17 = !{!18}
!18 = !DILocalVariable(name: "arg", arg: 1, scope: !7, file: !8, line: 3, type: !11)
!19 = !DILocation(line: 0, scope: !7)
!20 = !DILocation(line: 4, column: 5, scope: !7)
!21 = distinct !DISubprogram(name: "split_v4f32_multi_arg", scope: !8, file: !8, line: 7, type: !22, scopeLine: 7, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !28)
!22 = !DISubroutineType(types: !23)
!23 = !{!11, !11, !24}
!24 = !DIDerivedType(tag: DW_TAG_typedef, name: "float2", file: !12, line: 105, baseType: !25)
!25 = !DICompositeType(tag: DW_TAG_array_type, baseType: !14, size: 64, flags: DIFlagVector, elements: !26)
!26 = !{!27}
!27 = !DISubrange(count: 2)
!28 = !{!29, !30}
!29 = !DILocalVariable(name: "arg0", arg: 1, scope: !21, file: !8, line: 7, type: !11)
!30 = !DILocalVariable(name: "arg1", arg: 2, scope: !21, file: !8, line: 7, type: !24)
!31 = !DILocation(line: 0, scope: !21)
!32 = !DILocation(line: 8, column: 19, scope: !21)
!33 = !DILocation(line: 8, column: 17, scope: !21)
!34 = !DILocation(line: 8, column: 5, scope: !21)
!35 = distinct !DISubprogram(name: "split_v4f16_arg", scope: !8, file: !8, line: 11, type: !36, scopeLine: 11, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !41)
!36 = !DISubroutineType(types: !37)
!37 = !{!38, !38}
!38 = !DIDerivedType(tag: DW_TAG_typedef, name: "half4", file: !12, line: 114, baseType: !39)
!39 = !DICompositeType(tag: DW_TAG_array_type, baseType: !40, size: 64, flags: DIFlagVector, elements: !15)
!40 = !DIBasicType(name: "half", size: 16, encoding: DW_ATE_float)
!41 = !{!42}
!42 = !DILocalVariable(name: "arg", arg: 1, scope: !35, file: !8, line: 11, type: !38)
!43 = !DILocation(line: 0, scope: !35)
!44 = !DILocation(line: 12, column: 5, scope: !35)
!45 = distinct !DISubprogram(name: "split_f64_arg", scope: !8, file: !8, line: 15, type: !46, scopeLine: 15, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !49)
!46 = !DISubroutineType(types: !47)
!47 = !{!48, !48}
!48 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!49 = !{!50}
!50 = !DILocalVariable(name: "arg", arg: 1, scope: !45, file: !8, line: 15, type: !48)
!51 = !DILocation(line: 0, scope: !45)
!52 = !DILocation(line: 16, column: 5, scope: !45)
!53 = distinct !DISubprogram(name: "split_v2f64_arg", scope: !8, file: !8, line: 19, type: !54, scopeLine: 19, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !58)
!54 = !DISubroutineType(types: !55)
!55 = !{!56, !56}
!56 = !DIDerivedType(tag: DW_TAG_typedef, name: "double2", file: !12, line: 122, baseType: !57)
!57 = !DICompositeType(tag: DW_TAG_array_type, baseType: !48, size: 128, flags: DIFlagVector, elements: !26)
!58 = !{!59}
!59 = !DILocalVariable(name: "arg", arg: 1, scope: !53, file: !8, line: 19, type: !56)
!60 = !DILocation(line: 0, scope: !53)
!61 = !DILocation(line: 20, column: 5, scope: !53)
!62 = distinct !DISubprogram(name: "split_i64_arg", scope: !8, file: !8, line: 23, type: !63, scopeLine: 23, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !66)
!63 = !DISubroutineType(types: !64)
!64 = !{!65, !65}
!65 = !DIBasicType(name: "long int", size: 64, encoding: DW_ATE_signed)
!66 = !{!67}
!67 = !DILocalVariable(name: "arg", arg: 1, scope: !62, file: !8, line: 23, type: !65)
!68 = !DILocation(line: 0, scope: !62)
!69 = !DILocation(line: 24, column: 5, scope: !62)
!70 = distinct !DISubprogram(name: "split_ptr_arg", scope: !8, file: !8, line: 27, type: !71, scopeLine: 27, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !75)
!71 = !DISubroutineType(types: !72)
!72 = !{!73, !73}
!73 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !74, size: 64)
!74 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!75 = !{!76}
!76 = !DILocalVariable(name: "arg", arg: 1, scope: !70, file: !8, line: 27, type: !73)
!77 = !DILocation(line: 0, scope: !70)
!78 = !DILocation(line: 28, column: 5, scope: !70)
