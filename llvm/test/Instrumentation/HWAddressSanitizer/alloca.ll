; Test alloca instrumentation.
;
; RUN: opt < %s -passes=hwasan -hwasan-with-ifunc=1 -S | FileCheck %s --check-prefixes=CHECK,DYNAMIC-SHADOW,NO-UAR-TAGS
; RUN: opt < %s -passes=hwasan -hwasan-mapping-offset=0 -S | FileCheck %s --check-prefixes=CHECK,ZERO-BASED-SHADOW,NO-UAR-TAGS
; RUN: opt < %s -passes=hwasan -hwasan-with-ifunc=1 -hwasan-uar-retag-to-zero=0 -S | FileCheck %s --check-prefixes=CHECK,DYNAMIC-SHADOW,UAR-TAGS

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-android10000"

declare void @use32(ptr)

define void @test_alloca() sanitize_hwaddress !dbg !15 {
; CHECK-LABEL: @test_alloca(
; CHECK: %[[FP:[^ ]*]] = call ptr @llvm.frameaddress.p0(i32 0)
; CHECK: %[[A:[^ ]*]] = ptrtoint ptr %[[FP]] to i64
; CHECK: %[[B:[^ ]*]] = lshr i64 %[[A]], 20
; CHECK: %[[BASE_TAG:[^ ]*]] = xor i64 %[[A]], %[[B]]

; CHECK: %[[X:[^ ]*]] = alloca { i32, [12 x i8] }, align 16
; CHECK: %[[X_TAG:[^ ]*]] = xor i64 %[[BASE_TAG]], 0
; CHECK: %[[X1:[^ ]*]] = ptrtoint ptr %[[X]] to i64
; CHECK: %[[C:[^ ]*]] = shl i64 %[[X_TAG]], 56
; CHECK: %[[D:[^ ]*]] = or i64 %[[X1]], %[[C]]
; CHECK: %[[X_HWASAN:[^ ]*]] = inttoptr i64 %[[D]] to ptr

; CHECK: %[[X_TAG2:[^ ]*]] = trunc i64 %[[X_TAG]] to i8
; CHECK: %[[E:[^ ]*]] = ptrtoint ptr %[[X]] to i64
; CHECK: %[[F:[^ ]*]] = lshr i64 %[[E]], 4
; DYNAMIC-SHADOW: %[[X_SHADOW:[^ ]*]] = getelementptr i8, ptr %.hwasan.shadow, i64 %[[F]]
; ZERO-BASED-SHADOW: %[[X_SHADOW:[^ ]*]] = inttoptr i64 %[[F]] to ptr
; CHECK: %[[X_SHADOW_GEP:[^ ]*]] = getelementptr i8, ptr %[[X_SHADOW]], i32 0
; CHECK: store i8 4, ptr %[[X_SHADOW_GEP]]
; CHECK: %[[X_I8_GEP:[^ ]*]] = getelementptr i8, ptr %[[X]], i32 15
; CHECK: store i8 %[[X_TAG2]], ptr %[[X_I8_GEP]]
; CHECK: call void @llvm.dbg.value(
; CHECK-SAME: metadata !DIArgList(ptr %[[X]], ptr %[[X]])
; CHECK-SAME: metadata !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_tag_offset, 0, DW_OP_LLVM_arg, 1, DW_OP_LLVM_tag_offset, 0,
; CHECK: call void @use32(ptr nonnull %[[X_HWASAN]])

; UAR-TAGS: %[[BASE_TAG_COMPL:[^ ]*]] = xor i64 %[[BASE_TAG]], 255
; UAR-TAGS: %[[X_TAG_UAR:[^ ]*]] = trunc i64 %[[BASE_TAG_COMPL]] to i8
; CHECK: %[[E2:[^ ]*]] = ptrtoint ptr %[[X]] to i64
; CHECK: %[[F2:[^ ]*]] = lshr i64 %[[E2]], 4
; DYNAMIC-SHADOW: %[[X_SHADOW2:[^ ]*]] = getelementptr i8, ptr %.hwasan.shadow, i64 %[[F2]]
; ZERO-BASED-SHADOW: %[[X_SHADOW2:[^ ]*]] = inttoptr i64 %[[F2]] to ptr
; NO-UAR-TAGS: call void @llvm.memset.p0.i64(ptr align 1 %[[X_SHADOW2]], i8 0, i64 1, i1 false)
; UAR-TAGS: call void @llvm.memset.p0.i64(ptr align 1 %[[X_SHADOW2]], i8 %[[X_TAG_UAR]], i64 1, i1 false)
; CHECK: ret void


entry:
  %x = alloca i32, align 4
  call void @llvm.dbg.value(metadata !DIArgList(ptr %x, ptr %x), metadata !22, metadata !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_plus, DW_OP_deref)), !dbg !21
  call void @use32(ptr nonnull %x), !dbg !23
  ret void, !dbg !24
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!14}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 13.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "alloca.cpp", directory: "/")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!14 = !{!"clang version 13.0.0"}
!15 = distinct !DISubprogram(name: "test_alloca", linkageName: "_Z11test_allocav", scope: !1, file: !1, line: 4, type: !16, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!16 = !DISubroutineType(types: !17)
!17 = !{null}
!19 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !20, size: 64)
!20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!21 = !DILocation(line: 0, scope: !15)
!22 = !DILocalVariable(name: "x", scope: !15, file: !1, line: 5, type: !20)
!23 = !DILocation(line: 7, column: 5, scope: !15)
!24 = !DILocation(line: 8, column: 1, scope: !15)
