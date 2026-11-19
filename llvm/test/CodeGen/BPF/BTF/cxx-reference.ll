; RUN: llc -mtriple=bpfel -filetype=obj -o %t1 %s
; RUN: llvm-objcopy --dump-section='.BTF'=%t2 %t1
; RUN: %python %p/print_btf.py %t2 | FileCheck %s
; RUN: llc -mtriple=bpfeb -filetype=obj -o %t1 %s
; RUN: llvm-objcopy --dump-section='.BTF'=%t2 %t1
; RUN: %python %p/print_btf.py %t2 | FileCheck %s

; BTF has no references, so a C++ reference becomes a pointer, which has the
; same layout. Like a pointer, it leaves a record it refers to as a forward
; declaration, also when a second member uses the same reference type and the
; record has a complete definition in the debug info.
;
; struct Big { int x; Big *next; };
; struct R { int &r; int &&rr; int x; Big &b1; Big &b2; };
; int f(R *r) { return r->x; }
; Compilation flags:
;   clang --target=bpf -O2 -g -fstandalone-debug -S -emit-llvm test.cpp

; CHECK:      [1] PTR '(anon)' type_id=2
; CHECK-NEXT: [2] STRUCT 'R' size=40 vlen=5
; CHECK-NEXT:         'r' type_id=3 bits_offset=0
; CHECK-NEXT:         'rr' type_id=5 bits_offset=64
; CHECK-NEXT:         'x' type_id=4 bits_offset=128
; CHECK-NEXT:         'b1' type_id=6 bits_offset=192
; CHECK-NEXT:         'b2' type_id=6 bits_offset=256
; CHECK-NEXT: [3] PTR '(anon)' type_id=4
; CHECK-NEXT: [4] INT 'int' size=4 bits_offset=0 nr_bits=32 encoding=SIGNED
; CHECK-NEXT: [5] PTR '(anon)' type_id=4
; CHECK-NEXT: [6] PTR '(anon)' type_id=9
; CHECK-NEXT: [7] FUNC_PROTO '(anon)' ret_type_id=4 vlen=1
; CHECK-NEXT:         'r' type_id=1
; CHECK-NEXT: [8] FUNC 'f' type_id=7 linkage=global
; CHECK-NEXT: [9] FWD 'Big' fwd_kind=struct

target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "bpf"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read)
define dso_local noundef i32 @_Z1fP1R(ptr nofree noundef readonly captures(none) %r) local_unnamed_addr #0 !dbg !11 {
entry:
    #dbg_value(ptr %r, !33, !DIExpression(), !34)
  %x = getelementptr inbounds nuw i8, ptr %r, i64 16, !dbg !35
  %0 = load i32, ptr %x, align 8, !dbg !36, !tbaa !37
  ret i32 %0, !dbg !42
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
!11 = distinct !DISubprogram(name: "f", linkageName: "_Z1fP1R", scope: !12, file: !12, line: 3, type: !13, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !32, keyInstructions: true)
!12 = !DIFile(filename: "test.cpp", directory: "/")
!13 = !DISubroutineType(types: !14)
!14 = !{!15, !16}
!15 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !17, size: 64)
!17 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "R", file: !12, line: 2, size: 320, flags: DIFlagTypePassByValue, elements: !18, identifier: "_ZTS1R")
!18 = !{!19, !21, !23, !24, !31}
!19 = !DIDerivedType(tag: DW_TAG_member, name: "r", scope: !17, file: !12, line: 2, baseType: !20, size: 64)
!20 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !15, size: 64)
!21 = !DIDerivedType(tag: DW_TAG_member, name: "rr", scope: !17, file: !12, line: 2, baseType: !22, size: 64, offset: 64)
!22 = !DIDerivedType(tag: DW_TAG_rvalue_reference_type, baseType: !15, size: 64)
!23 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !17, file: !12, line: 2, baseType: !15, size: 32, offset: 128)
!24 = !DIDerivedType(tag: DW_TAG_member, name: "b1", scope: !17, file: !12, line: 2, baseType: !25, size: 64, offset: 192)
!25 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !26, size: 64)
!26 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Big", file: !12, line: 1, size: 128, flags: DIFlagTypePassByValue, elements: !27, identifier: "_ZTS3Big")
!27 = !{!28, !29}
!28 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !26, file: !12, line: 1, baseType: !15, size: 32)
!29 = !DIDerivedType(tag: DW_TAG_member, name: "next", scope: !26, file: !12, line: 1, baseType: !30, size: 64, offset: 64)
!30 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !26, size: 64)
!31 = !DIDerivedType(tag: DW_TAG_member, name: "b2", scope: !17, file: !12, line: 2, baseType: !25, size: 64, offset: 256)
!32 = !{!33}
!33 = !DILocalVariable(name: "r", arg: 1, scope: !11, file: !12, line: 3, type: !16)
!34 = !DILocation(line: 0, scope: !11)
!35 = !DILocation(line: 3, column: 25, scope: !11)
!36 = !DILocation(line: 3, column: 25, scope: !11, atomGroup: 1, atomRank: 2)
!37 = !{!38, !8, i64 16}
!38 = !{!"_ZTS1R", !39, i64 0, !39, i64 8, !8, i64 16, !41, i64 24, !41, i64 32}
!39 = !{!"p1 int", !40, i64 0}
!40 = !{!"any pointer", !9, i64 0}
!41 = !{!"p1 _ZTS3Big", !40, i64 0}
!42 = !DILocation(line: 3, column: 15, scope: !11, atomGroup: 1, atomRank: 1)
