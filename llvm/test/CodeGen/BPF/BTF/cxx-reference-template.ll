; RUN: llc -mtriple=bpfel -filetype=obj -o %t1 %s
; RUN: llvm-objcopy --dump-section='.BTF'=%t2 %t1
; RUN: %python %p/print_btf.py %t2 | FileCheck %s
; RUN: llc -mtriple=bpfeb -filetype=obj -o %t1 %s
; RUN: llvm-objcopy --dump-section='.BTF'=%t2 %t1
; RUN: %python %p/print_btf.py %t2 | FileCheck %s

; BTF has no references, so a C++ reference becomes a pointer, which has the
; same layout. The name of a template specialization, such as 'TC<int>' or an
; alias template, is emitted without its template arguments, as with
; -gsimple-template-names: released kernels only accept identifiers as names.
;
; template <class T> struct TC { T v; };
; template <class T> using Alias = TC<T>;
; template <class T> struct TF;
; struct R { int &r; int &&rr; int x; };
; int f(TC<int> *a, TC<long> *b, Alias<char> *c, TF<int> *d, R *r) {
;   return a->v + b->v + c->v + (d != 0) + r->x;
; }
; Compilation flags:
;   clang --target=bpf -O2 -g -S -emit-llvm test.cpp

; CHECK:      [1] PTR '(anon)' type_id=2
; CHECK-NEXT: [2] STRUCT 'TC' size=4 vlen=1
; CHECK-NEXT:         'v' type_id=3 bits_offset=0
; CHECK-NEXT: [3] INT 'int' size=4 bits_offset=0 nr_bits=32 encoding=SIGNED
; CHECK-NEXT: [4] PTR '(anon)' type_id=5
; CHECK-NEXT: [5] STRUCT 'TC' size=8 vlen=1
; CHECK-NEXT:         'v' type_id=6 bits_offset=0
; CHECK-NEXT: [6] INT 'long' size=8 bits_offset=0 nr_bits=64 encoding=SIGNED
; CHECK-NEXT: [7] PTR '(anon)' type_id=8
; CHECK-NEXT: [8] TYPEDEF 'Alias' type_id=9
; CHECK-NEXT: [9] STRUCT 'TC' size=1 vlen=1
; CHECK-NEXT:         'v' type_id=10 bits_offset=0
; CHECK-NEXT: [10] INT 'char' size=1 bits_offset=0 nr_bits=8 encoding=SIGNED
; CHECK-NEXT: [11] PTR '(anon)' type_id=12
; CHECK-NEXT: [12] FWD 'TF' fwd_kind=struct
; CHECK-NEXT: [13] PTR '(anon)' type_id=14
; CHECK-NEXT: [14] STRUCT 'R' size=24 vlen=3
; CHECK-NEXT:         'r' type_id=15 bits_offset=0
; CHECK-NEXT:         'rr' type_id=16 bits_offset=64
; CHECK-NEXT:         'x' type_id=3 bits_offset=128
; CHECK-NEXT: [15] PTR '(anon)' type_id=3
; CHECK-NEXT: [16] PTR '(anon)' type_id=3

target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "bpf"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read)
define dso_local noundef i32 @_Z1fP2TCIiEPS_IlEPS_IcEP2TFIiEP1R(ptr nofree noundef readonly captures(none) %a, ptr nofree noundef readonly captures(none) %b, ptr nofree noundef readonly captures(none) %c, ptr nofree noundef readnone captures(address_is_null) %d, ptr nofree noundef readonly captures(none) %r) local_unnamed_addr #0 !dbg !11 {
entry:
    #dbg_value(ptr %a, !47, !DIExpression(), !52)
    #dbg_value(ptr %b, !48, !DIExpression(), !52)
    #dbg_value(ptr %c, !49, !DIExpression(), !52)
    #dbg_value(ptr %d, !50, !DIExpression(), !52)
    #dbg_value(ptr %r, !51, !DIExpression(), !52)
  %0 = load i32, ptr %a, align 4, !dbg !53, !tbaa !54
  %conv = zext i32 %0 to i64, !dbg !56
  %1 = load i64, ptr %b, align 8, !dbg !57, !tbaa !58
  %2 = load i8, ptr %c, align 1, !dbg !61, !tbaa !62
  %conv3 = sext i8 %2 to i64, !dbg !64
  %cmp = icmp ne ptr %d, null, !dbg !65
  %conv5 = zext i1 %cmp to i64, !dbg !66
  %add = add nuw nsw i64 %conv, %conv5, !dbg !67
  %add4 = add i64 %add, %1, !dbg !68
  %add6 = add i64 %add4, %conv3, !dbg !69
  %x = getelementptr inbounds nuw i8, ptr %r, i64 16, !dbg !70
  %3 = load i32, ptr %x, align 8, !dbg !70, !tbaa !71
  %4 = trunc i64 %add6 to i32, !dbg !75
  %conv9 = add i32 %3, %4, !dbg !75
  ret i32 %conv9, !dbg !76
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5}
!llvm.errno.tbaa = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 7, !"frame-pointer", i32 2}
!5 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!6 = !{!7, !8, i64 0}
!7 = !{!"__libc_errno", !8, i64 0}
!8 = !{!"int", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C++ TBAA"}
!11 = distinct !DISubprogram(name: "f", linkageName: "_Z1fP2TCIiEPS_IlEPS_IcEP2TFIiEP1R", scope: !1, file: !1, line: 5, type: !12, scopeLine: 5, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !46, keyInstructions: true)
!12 = !DISubroutineType(types: !13)
!13 = !{!14, !15, !21, !28, !36, !38}
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !16, size: 64)
!16 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "TC<int>", file: !1, line: 1, size: 32, flags: DIFlagTypePassByValue, elements: !17, templateParams: !19, identifier: "_ZTS2TCIiE")
!17 = !{!18}
!18 = !DIDerivedType(tag: DW_TAG_member, name: "v", scope: !16, file: !1, line: 1, baseType: !14, size: 32)
!19 = !{!20}
!20 = !DITemplateTypeParameter(name: "T", type: !14)
!21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !22, size: 64)
!22 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "TC<long>", file: !1, line: 1, size: 64, flags: DIFlagTypePassByValue, elements: !23, templateParams: !26, identifier: "_ZTS2TCIlE")
!23 = !{!24}
!24 = !DIDerivedType(tag: DW_TAG_member, name: "v", scope: !22, file: !1, line: 1, baseType: !25, size: 64)
!25 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!26 = !{!27}
!27 = !DITemplateTypeParameter(name: "T", type: !25)
!28 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !29, size: 64)
!29 = !DIDerivedType(tag: DW_TAG_typedef, name: "Alias<char>", file: !1, line: 2, baseType: !30)
!30 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "TC<char>", file: !1, line: 1, size: 8, flags: DIFlagTypePassByValue, elements: !31, templateParams: !34, identifier: "_ZTS2TCIcE")
!31 = !{!32}
!32 = !DIDerivedType(tag: DW_TAG_member, name: "v", scope: !30, file: !1, line: 1, baseType: !33, size: 8)
!33 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!34 = !{!35}
!35 = !DITemplateTypeParameter(name: "T", type: !33)
!36 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !37, size: 64)
!37 = !DICompositeType(tag: DW_TAG_structure_type, name: "TF<int>", file: !1, line: 3, flags: DIFlagFwdDecl | DIFlagNonTrivial, identifier: "_ZTS2TFIiE")
!38 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !39, size: 64)
!39 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "R", file: !1, line: 4, size: 192, flags: DIFlagTypePassByValue, elements: !40, identifier: "_ZTS1R")
!40 = !{!41, !43, !45}
!41 = !DIDerivedType(tag: DW_TAG_member, name: "r", scope: !39, file: !1, line: 4, baseType: !42, size: 64)
!42 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !14, size: 64)
!43 = !DIDerivedType(tag: DW_TAG_member, name: "rr", scope: !39, file: !1, line: 4, baseType: !44, size: 64, offset: 64)
!44 = !DIDerivedType(tag: DW_TAG_rvalue_reference_type, baseType: !14, size: 64)
!45 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !39, file: !1, line: 4, baseType: !14, size: 32, offset: 128)
!46 = !{!47, !48, !49, !50, !51}
!47 = !DILocalVariable(name: "a", arg: 1, scope: !11, file: !1, line: 5, type: !15)
!48 = !DILocalVariable(name: "b", arg: 2, scope: !11, file: !1, line: 5, type: !21)
!49 = !DILocalVariable(name: "c", arg: 3, scope: !11, file: !1, line: 5, type: !28)
!50 = !DILocalVariable(name: "d", arg: 4, scope: !11, file: !1, line: 5, type: !36)
!51 = !DILocalVariable(name: "r", arg: 5, scope: !11, file: !1, line: 5, type: !38)
!52 = !DILocation(line: 0, scope: !11)
!53 = !DILocation(line: 6, column: 13, scope: !11)
!54 = !{!55, !8, i64 0}
!55 = !{!"_ZTS2TCIiE", !8, i64 0}
!56 = !DILocation(line: 6, column: 10, scope: !11)
!57 = !DILocation(line: 6, column: 20, scope: !11)
!58 = !{!59, !60, i64 0}
!59 = !{!"_ZTS2TCIlE", !60, i64 0}
!60 = !{!"long", !9, i64 0}
!61 = !DILocation(line: 6, column: 27, scope: !11)
!62 = !{!63, !9, i64 0}
!63 = !{!"_ZTS2TCIcE", !9, i64 0}
!64 = !DILocation(line: 6, column: 24, scope: !11)
!65 = !DILocation(line: 6, column: 34, scope: !11)
!66 = !DILocation(line: 6, column: 31, scope: !11)
!67 = !DILocation(line: 6, column: 15, scope: !11)
!68 = !DILocation(line: 6, column: 22, scope: !11)
!69 = !DILocation(line: 6, column: 29, scope: !11)
!70 = !DILocation(line: 6, column: 45, scope: !11)
!71 = !{!72, !8, i64 16}
!72 = !{!"_ZTS1R", !73, i64 0, !73, i64 8, !8, i64 16}
!73 = !{!"p1 int", !74, i64 0}
!74 = !{!"any pointer", !9, i64 0}
!75 = !DILocation(line: 6, column: 10, scope: !11, atomGroup: 1, atomRank: 2)
!76 = !DILocation(line: 6, column: 3, scope: !11, atomGroup: 1, atomRank: 1)
