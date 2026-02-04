; RUN: llc < %s -O0 --filetype=obj -o - | llvm-dwarfdump - | FileCheck %s --check-prefixes DBG_USE

target triple = "wasm32-unknown-unknown"

declare i32 @extern_func(i32, i32)

; We want to produce WASM local "DW_OP_WASM_location 0x00 <local number>" locations
; in debug code instead of operand stack ("DW_OP_WASM_location 0x02 <depth>") locations
; since local locations are more widely supported and can cover the entirety of the method.
; DBG_USE: DW_TAG_subprogram
; DBG_USE:   DW_AT_name ("single_non_dbg_use")
; DBG_USE:   DW_TAG_variable
; DBG_USE:     DW_AT_location
; DBG_USE:       DW_OP_WASM_location 0x0
; DBG_USE:     DW_AT_name    ("call_value")
; DBG_USE:   DW_TAG_variable
; DBG_USE:     DW_AT_location
; DBG_USE:       DW_OP_WASM_location 0x0
; DBG_USE:     DW_AT_name    ("sub_value")
define i32 @single_non_dbg_use(i32 %0, i32 %1) !dbg !6 {
  %call_value = call i32 @extern_func(i32 1, i32 2), !dbg !20
  call void @llvm.dbg.value(metadata i32 %call_value, metadata !11, metadata !DIExpression()), !dbg !20
  %div = udiv i32 %0, %1, !dbg !21
  %sub = sub i32 %call_value, %div, !dbg !22
  call void @llvm.dbg.value(metadata i32 %sub, metadata !12, metadata !DIExpression()), !dbg !22
  ret i32 %sub, !dbg !23
}

!6 = distinct !DISubprogram(name: "single_non_dbg_use", scope: !1, file: !1, type: !7, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!7 = !DISubroutineType(types: !8)
!8 = !{!9, !9, !9}
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DILocalVariable(name: "call_value", scope: !6, type: !9)
!12 = !DILocalVariable(name: "sub_value", scope: !6, type: !9)
!20 = !DILocation(line: 20, scope: !6)
!21 = !DILocation(line: 21, scope: !6)
!22 = !DILocation(line: 22, scope: !6)
!23 = !DILocation(line: 23, scope: !6)

; Similarly for a singly-used frame base.
; DBG_USE: DW_TAG_subprogram
; DBG_USE:   DW_AT_frame_base (DW_OP_WASM_location 0x0
; DBG_USE:   DW_AT_name ("single_use_frame_base")
; DBG_USE:   DW_TAG_variable
; DBG_USE:     DW_AT_location (DW_OP_fbreg +12)
; DBG_USE:     DW_AT_name ("arg_value")
define i32 @single_use_frame_base(i32 %0, i32 %1) !dbg !13 {
  %arg_loc = alloca i32, !dbg !24
  store i32 %1, ptr %arg_loc, !dbg !25
  call void @llvm.dbg.declare(metadata ptr %arg_loc, metadata !14, metadata !DIExpression()), !dbg !25
  ret i32 %0, !dbg !26
}

!13 = distinct !DISubprogram(name: "single_use_frame_base", scope: !1, file: !1, type: !7, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!14 = !DILocalVariable(name: "arg_value", scope: !13, type: !9)
!24 = !DILocation(line: 24, scope: !13)
!25 = !DILocation(line: 25, scope: !13)
!26 = !DILocation(line: 26, scope: !13)

; Similarly for multivalue defs... But we can't really test
; it due to https://github.com/llvm/llvm-project/issues/136506.
; declare {i32, i32} @extern_func_multivalue(i32, i32)
; 
; define i32 @single_non_dbg_use_multivalue(i32 %0, i32 %1) !dbg !15 {
;   %full_value = call {i32, i32} @extern_func_multivalue(i32 1, i32 2), !dbg !27
;   %full_value_one = extractvalue {i32, i32} %full_value, 0, !dbg !27
;   %full_value_two = extractvalue {i32, i32} %full_value, 1, !dbg !27
;   %partial_value = call {i32, i32} @extern_func_multivalue(i32 %full_value_one, i32 %full_value_two), !dbg !28
;   call void @llvm.dbg.value(metadata i32 %full_value_two, metadata !16, metadata !DIExpression()), !dbg !28
;   %partial_value_one = extractvalue {i32, i32} %partial_value, 0, !dbg !28
;   %partial_value_two = extractvalue {i32, i32} %partial_value, 1, !dbg !28
;   call void @llvm.dbg.value(metadata i32 %partial_value_two, metadata !17, metadata !DIExpression()), !dbg !28
;   ret i32 %partial_value_one, !dbg !29
; }
; 
; !15 = distinct !DISubprogram(name: "single_non_dbg_use_multivalue", scope: !1, file: !1, type: !7, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
; !16 = !DILocalVariable(name: "value_used", scope: !15, type: !9)
; !17 = !DILocalVariable(name: "value_unused", scope: !15, type: !9)
; !27 = !DILocation(line: 27, scope: !15)
; !28 = !DILocation(line: 28, scope: !15)
; !29 = !DILocation(line: 29, scope: !15)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "LLC", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.ll", directory: "")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
