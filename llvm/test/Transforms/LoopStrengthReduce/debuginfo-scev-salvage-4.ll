; RUN: opt -S -loop-reduce %s -o - | FileCheck %s
; RUN: opt --try-experimental-debuginfo-iterators -S -loop-reduce %s -o - | FileCheck %s
; REQUIRES: x86-registered-target

;; Ensure that we retain debuginfo for the induction variable and dependant
;; variables when loop strength reduction is applied to the loop. This test
;; covers the translation of a SCEVCastExpr to DIExpression containg 
;; DW_OP_LLVM_CONVERT. This IR produced from:
;;
;; clang -S -emit-llvm -Xclang -disable-llvm-passes -g lsr-basic.cpp -o
;; Then executing opt -O2 up to the the loopFullUnroll pass.
;; void mul_pow_of_2_to_shift(unsigned size, unsigned *data) {
;; 
;; void zext_scev(int64_t *arr, uint32_t factor0, int16_t factor1) {
;;     uint32_t i = 0;
;;     while(i < 63) {
;;         uint32_t comp = factor0 - (4*i*factor1);
;;         arr[i] = comp;
;;         ++i;
;;     }
;; }
; CHECK: call void @llvm.dbg.value(metadata i64 %lsr.iv, metadata ![[i:[0-9]+]], metadata !DIExpression())
; CHECK: call void @llvm.dbg.value(metadata !DIArgList(i64 %lsr.iv, i16 %factor1, i32 %factor0), metadata ![[comp:[0-9]+]], metadata !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_consts, 18446744073709551612, DW_OP_LLVM_arg, 1, DW_OP_LLVM_convert, 32, DW_ATE_signed, DW_OP_mul, DW_OP_mul, DW_OP_LLVM_arg, 2, DW_OP_plus, DW_OP_stack_value))
; CHECK: call void @llvm.dbg.value(metadata i64 %lsr.iv, metadata ![[i]], metadata !DIExpression(DW_OP_consts, 1, DW_OP_plus, DW_OP_stack_value))
; CHECK: ![[i]] = !DILocalVariable(name: "i"
; CHECK: ![[comp]] = !DILocalVariable(name: "comp"

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define dso_local void @_Z9zext_scevPljs(ptr nocapture %arr, i32 %factor0, i16 signext %factor1) local_unnamed_addr !dbg !90 {
entry:
  call void @llvm.dbg.value(metadata ptr %arr, metadata !94, metadata !DIExpression()), !dbg !95
  call void @llvm.dbg.value(metadata i32 %factor0, metadata !96, metadata !DIExpression()), !dbg !95
  call void @llvm.dbg.value(metadata i16 %factor1, metadata !97, metadata !DIExpression()), !dbg !95
  call void @llvm.dbg.value(metadata i32 0, metadata !98, metadata !DIExpression()), !dbg !95
  %conv = sext i16 %factor1 to i32
  %mul.neg = mul i32 %conv, -4
  call void @llvm.dbg.value(metadata i32 0, metadata !98, metadata !DIExpression()), !dbg !95
  br label %while.body, !dbg !95

while.body:                                       ; preds = %while.body, %entry
  %i.04 = phi i32 [ 0, %entry ], [ %inc, %while.body ]
  call void @llvm.dbg.value(metadata i32 %i.04, metadata !98, metadata !DIExpression()), !dbg !95
  %mul1.neg = mul i32 %mul.neg, %i.04, !dbg !99
  %sub = add i32 %mul1.neg, %factor0, !dbg !99
  call void @llvm.dbg.value(metadata i32 %sub, metadata !101, metadata !DIExpression()), !dbg !99
  %conv2 = zext i32 %sub to i64, !dbg !99
  %idxprom = zext i32 %i.04 to i64, !dbg !99
  %arrayidx = getelementptr inbounds i64, ptr %arr, i64 %idxprom, !dbg !99
  store i64 %conv2, ptr %arrayidx, align 8, !dbg !99
  %inc = add nuw nsw i32 %i.04, 1, !dbg !99
  call void @llvm.dbg.value(metadata i32 %inc, metadata !98, metadata !DIExpression()), !dbg !95
  %cmp = icmp ult i32 %inc, 63, !dbg !95
  br i1 %cmp, label %while.body, label %while.end, !dbg !95, !llvm.loop !102

while.end:                                        ; preds = %while.body
  ret void, !dbg !95
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

attributes #0 = { nofree nosync nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!84, !85, !86, !87, !88}
!llvm.ident = !{!89}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 13.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, imports: !3, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "cast.cpp", directory: "/test")
!2 = !{}
!3 = !{!4, !12, !16, !20, !24, !27, !29, !31, !33, !35, !37, !39, !41, !44, !46, !51, !55, !59, !63, !65, !67, !69, !71, !73, !75, !77, !79, !82}
!4 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !6, file: !11, line: 48)
!5 = !DINamespace(name: "std", scope: null)
!6 = !DIDerivedType(tag: DW_TAG_typedef, name: "int8_t", file: !7, line: 24, baseType: !8)
!7 = !DIFile(filename: "/usr/include/x86_64-linux-gnu/bits/stdint-intn.h", directory: "")
!8 = !DIDerivedType(tag: DW_TAG_typedef, name: "__int8_t", file: !9, line: 36, baseType: !10)
!9 = !DIFile(filename: "/usr/include/x86_64-linux-gnu/bits/types.h", directory: "")
!10 = !DIBasicType(name: "signed char", size: 8, encoding: DW_ATE_signed_char)
!11 = !DIFile(filename: "/usr/lib/gcc/x86_64-linux-gnu/8/../../../../include/c++/8/cstdint", directory: "")
!12 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !13, file: !11, line: 49)
!13 = !DIDerivedType(tag: DW_TAG_typedef, name: "int16_t", file: !7, line: 25, baseType: !14)
!14 = !DIDerivedType(tag: DW_TAG_typedef, name: "__int16_t", file: !9, line: 38, baseType: !15)
!15 = !DIBasicType(name: "short", size: 16, encoding: DW_ATE_signed)
!16 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !17, file: !11, line: 50)
!17 = !DIDerivedType(tag: DW_TAG_typedef, name: "int32_t", file: !7, line: 26, baseType: !18)
!18 = !DIDerivedType(tag: DW_TAG_typedef, name: "__int32_t", file: !9, line: 40, baseType: !19)
!19 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!20 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !21, file: !11, line: 51)
!21 = !DIDerivedType(tag: DW_TAG_typedef, name: "int64_t", file: !7, line: 27, baseType: !22)
!22 = !DIDerivedType(tag: DW_TAG_typedef, name: "__int64_t", file: !9, line: 43, baseType: !23)
!23 = !DIBasicType(name: "long int", size: 64, encoding: DW_ATE_signed)
!24 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !25, file: !11, line: 53)
!25 = !DIDerivedType(tag: DW_TAG_typedef, name: "int_fast8_t", file: !26, line: 68, baseType: !10)
!26 = !DIFile(filename: "/usr/include/stdint.h", directory: "")
!27 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !28, file: !11, line: 54)
!28 = !DIDerivedType(tag: DW_TAG_typedef, name: "int_fast16_t", file: !26, line: 70, baseType: !23)
!29 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !30, file: !11, line: 55)
!30 = !DIDerivedType(tag: DW_TAG_typedef, name: "int_fast32_t", file: !26, line: 71, baseType: !23)
!31 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !32, file: !11, line: 56)
!32 = !DIDerivedType(tag: DW_TAG_typedef, name: "int_fast64_t", file: !26, line: 72, baseType: !23)
!33 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !34, file: !11, line: 58)
!34 = !DIDerivedType(tag: DW_TAG_typedef, name: "int_least8_t", file: !26, line: 43, baseType: !10)
!35 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !36, file: !11, line: 59)
!36 = !DIDerivedType(tag: DW_TAG_typedef, name: "int_least16_t", file: !26, line: 44, baseType: !15)
!37 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !38, file: !11, line: 60)
!38 = !DIDerivedType(tag: DW_TAG_typedef, name: "int_least32_t", file: !26, line: 45, baseType: !19)
!39 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !40, file: !11, line: 61)
!40 = !DIDerivedType(tag: DW_TAG_typedef, name: "int_least64_t", file: !26, line: 47, baseType: !23)
!41 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !42, file: !11, line: 63)
!42 = !DIDerivedType(tag: DW_TAG_typedef, name: "intmax_t", file: !26, line: 111, baseType: !43)
!43 = !DIDerivedType(tag: DW_TAG_typedef, name: "__intmax_t", file: !9, line: 61, baseType: !23)
!44 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !45, file: !11, line: 64)
!45 = !DIDerivedType(tag: DW_TAG_typedef, name: "intptr_t", file: !26, line: 97, baseType: !23)
!46 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !47, file: !11, line: 66)
!47 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint8_t", file: !48, line: 24, baseType: !49)
!48 = !DIFile(filename: "/usr/include/x86_64-linux-gnu/bits/stdint-uintn.h", directory: "")
!49 = !DIDerivedType(tag: DW_TAG_typedef, name: "__uint8_t", file: !9, line: 37, baseType: !50)
!50 = !DIBasicType(name: "unsigned char", size: 8, encoding: DW_ATE_unsigned_char)
!51 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !52, file: !11, line: 67)
!52 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint16_t", file: !48, line: 25, baseType: !53)
!53 = !DIDerivedType(tag: DW_TAG_typedef, name: "__uint16_t", file: !9, line: 39, baseType: !54)
!54 = !DIBasicType(name: "unsigned short", size: 16, encoding: DW_ATE_unsigned)
!55 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !56, file: !11, line: 68)
!56 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint32_t", file: !48, line: 26, baseType: !57)
!57 = !DIDerivedType(tag: DW_TAG_typedef, name: "__uint32_t", file: !9, line: 41, baseType: !58)
!58 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!59 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !60, file: !11, line: 69)
!60 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint64_t", file: !48, line: 27, baseType: !61)
!61 = !DIDerivedType(tag: DW_TAG_typedef, name: "__uint64_t", file: !9, line: 44, baseType: !62)
!62 = !DIBasicType(name: "long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!63 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !64, file: !11, line: 71)
!64 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint_fast8_t", file: !26, line: 81, baseType: !50)
!65 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !66, file: !11, line: 72)
!66 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint_fast16_t", file: !26, line: 83, baseType: !62)
!67 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !68, file: !11, line: 73)
!68 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint_fast32_t", file: !26, line: 84, baseType: !62)
!69 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !70, file: !11, line: 74)
!70 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint_fast64_t", file: !26, line: 85, baseType: !62)
!71 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !72, file: !11, line: 76)
!72 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint_least8_t", file: !26, line: 54, baseType: !50)
!73 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !74, file: !11, line: 77)
!74 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint_least16_t", file: !26, line: 55, baseType: !54)
!75 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !76, file: !11, line: 78)
!76 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint_least32_t", file: !26, line: 56, baseType: !58)
!77 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !78, file: !11, line: 79)
!78 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint_least64_t", file: !26, line: 58, baseType: !62)
!79 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !80, file: !11, line: 81)
!80 = !DIDerivedType(tag: DW_TAG_typedef, name: "uintmax_t", file: !26, line: 112, baseType: !81)
!81 = !DIDerivedType(tag: DW_TAG_typedef, name: "__uintmax_t", file: !9, line: 62, baseType: !62)
!82 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !83, file: !11, line: 82)
!83 = !DIDerivedType(tag: DW_TAG_typedef, name: "uintptr_t", file: !26, line: 100, baseType: !62)
!84 = !{i32 7, !"Dwarf Version", i32 4}
!85 = !{i32 2, !"Debug Info Version", i32 3}
!86 = !{i32 1, !"wchar_size", i32 4}
!87 = !{i32 7, !"uwtable", i32 1}
!88 = !{i32 7, !"frame-pointer", i32 2}
!89 = !{!"clang version 13.0.0"}
!90 = distinct !DISubprogram(name: "zext_scev", linkageName: "_Z9zext_scevPljs", scope: !1, file: !1, line: 4, type: !91, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!91 = !DISubroutineType(types: !92)
!92 = !{null, !93, !56, !13}
!93 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !21, size: 64)
!94 = !DILocalVariable(name: "arr", arg: 1, scope: !90, file: !1, line: 4, type: !93)
!95 = !DILocation(line: 0, scope: !90)
!96 = !DILocalVariable(name: "factor0", arg: 2, scope: !90, file: !1, line: 4, type: !56)
!97 = !DILocalVariable(name: "factor1", arg: 3, scope: !90, file: !1, line: 4, type: !13)
!98 = !DILocalVariable(name: "i", scope: !90, file: !1, line: 5, type: !56)
!99 = !DILocation(line: 7, column: 39, scope: !100)
!100 = distinct !DILexicalBlock(scope: !90, file: !1, line: 6, column: 19)
!101 = !DILocalVariable(name: "comp", scope: !100, file: !1, line: 7, type: !56)
!102 = distinct !{!102, !103, !104, !105}
!103 = !DILocation(line: 6, column: 5, scope: !90)
!104 = !DILocation(line: 10, column: 5, scope: !90)
!105 = !{!"llvm.loop.mustprogress"}
