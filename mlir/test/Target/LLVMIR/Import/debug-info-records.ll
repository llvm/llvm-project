; RUN: mlir-translate -import-llvm -mlir-print-debuginfo -convert-debug-rec-to-intrinsics -emit-expensive-warnings -split-input-file %s 2>&1 | FileCheck %s
; RUN: mlir-translate -import-llvm -mlir-print-debuginfo -emit-expensive-warnings -split-input-file %s 2>&1 | FileCheck %s
; XFAIL: *
; CHECK: #[[LOCAL_VAR0:.*]] = #llvm.di_local_variable<scope = #di_lexical_block>
; CHECK: #[[LOCAL_VAR1:.*]] = #llvm.di_local_variable<scope = #di_lexical_block_file, name = "arg"
; CHECK: #[[LOCAL_VAR2:.*]] = #llvm.di_local_variable<scope = #di_lexical_block, name = "alloc"

; CHECK: @callee()
define void @callee() {
  ret void
}

define void @func_with_empty_named_info() {
  call void @callee()
  ret void
}

define void @func_no_debug() {
  ret void
}

; CHECK: llvm.func @func_with_debug(%[[ARG0:.*]]: i64
define void @func_with_debug(i64 %0) !dbg !3 {

  ; CHECK: llvm.intr.dbg.value #[[LOCAL_VAR0]] = %[[ARG0]] : i64
  ; CHECK: llvm.intr.dbg.value #[[LOCAL_VAR1]] #llvm.di_expression<[DW_OP_LLVM_fragment(0, 1)]> = %[[ARG0]] : i64
  ; CHECK: %[[CST:.*]] = llvm.mlir.constant(1 : i32) : i32
  ; CHECK: %[[ADDR:.*]] = llvm.alloca %[[CST]] x i64
  ; CHECK: llvm.intr.dbg.declare #[[LOCAL_VAR2]] #llvm.di_expression<[DW_OP_deref, DW_OP_LLVM_convert(4, DW_ATE_signed)]> = %[[ADDR]] : !llvm.ptr
  %2 = alloca i64, align 8, !dbg !19
    #dbg_value(i64 %0, !20, !DIExpression(DW_OP_LLVM_fragment, 0, 1), !22)
    #dbg_declare(ptr %2, !23, !DIExpression(DW_OP_deref, DW_OP_LLVM_convert, 4, DW_ATE_signed), !25)
    #dbg_value(i64 %0, !26, !DIExpression(), !27)
  call void @func_no_debug(), !dbg !28
  %3 = add i64 %0, %0, !dbg !32
  ret void, !dbg !37
}

define void @empty_types() !dbg !38 {
  ret void, !dbg !44
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "MLIR", isOptimized: true, runtimeVersion: 0, splitDebugFilename: "test.dwo", emissionKind: FullDebug, nameTableKind: None)
!1 = !DIFile(filename: "foo.mlir", directory: "/test/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "func_with_debug", linkageName: "func_with_debug", scope: !4, file: !1, line: 3, type: !6, scopeLine: 3, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!4 = !DINamespace(name: "nested", scope: !5)
!5 = !DINamespace(name: "toplevel", scope: null, exportSymbols: true)
!6 = !DISubroutineType(cc: DW_CC_normal, types: !7)
!7 = !{null, !8, !9, !11, !12, !13, !16}
!8 = !DIBasicType(name: "si64")
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64, align: 32, offset: 8, extraData: !10)
!10 = !DIBasicType(name: "si32", size: 32, encoding: DW_ATE_signed)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "named", baseType: !10)
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64, align: 32, offset: 8, dwarfAddressSpace: 3)
!13 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "composite", file: !1, line: 42, size: 64, align: 32, elements: !14)
!14 = !{!15}
!15 = !DISubrange(count: 4)
!16 = !DICompositeType(tag: DW_TAG_array_type, name: "array", file: !1, baseType: !8, flags: DIFlagVector, elements: !17)
!17 = !{!18}
!18 = !DISubrange(lowerBound: 0, upperBound: 4, stride: 1)
!19 = !DILocation(line: 100, column: 12, scope: !3)
!20 = !DILocalVariable(name: "arg", arg: 1, scope: !21, file: !1, line: 6, type: !8, align: 32)
!21 = distinct !DILexicalBlockFile(scope: !3, file: !1, discriminator: 0)
!22 = !DILocation(line: 103, column: 3, scope: !3)
!23 = !DILocalVariable(name: "alloc", scope: !24)
!24 = distinct !DILexicalBlock(scope: !3)
!25 = !DILocation(line: 106, column: 3, scope: !3)
!26 = !DILocalVariable(scope: !24)
!27 = !DILocation(line: 109, column: 3, scope: !3)
!28 = !DILocation(line: 1, column: 2, scope: !3)
!32 = !DILocation(line: 2, column: 4, scope: !33, inlinedAt: !36)
!33 = distinct !DISubprogram(name: "callee", scope: !13, file: !1, type: !34, spFlags: DISPFlagDefinition, unit: !0)
!34 = !DISubroutineType(types: !35)
!35 = !{!8, !8}
!36 = !DILocation(line: 28, column: 5, scope: !3)
!37 = !DILocation(line: 135, column: 3, scope: !3)
!38 = distinct !DISubprogram(name: "empty_types", scope: !39, file: !1, type: !40, spFlags: DISPFlagDefinition, unit: !0, annotations: !42)
!39 = !DIModule(scope: !1, name: "module", configMacros: "bar", includePath: "/", apinotes: "/", file: !1, line: 42, isDecl: true)
!40 = !DISubroutineType(cc: DW_CC_normal, types: !41)
!41 = !{}
!42 = !{!43}
!43 = !{!"foo", !"bar"}
!44 = !DILocation(line: 140, column: 3, scope: !38)
