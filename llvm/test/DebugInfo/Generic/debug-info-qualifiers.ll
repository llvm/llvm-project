; REQUIRES: object-emission
; Test (r)value qualifiers on C++11 non-static member functions.
; Generated from tools/clang/test/CodeGenCXX/debug-info-qualifiers.cpp
;
; class A {
; public:
;   void l() const &;
;   void r() const &&;
; };
;
; void g() {
;   A a;
;   auto pl = &A::l;
;   auto pr = &A::r;
; }
;
; RUN: %llc_dwarf -filetype=obj -O0 < %s | llvm-dwarfdump -v - | FileCheck %s
; CHECK: DW_TAG_subroutine_type     DW_CHILDREN_yes
; CHECK-NEXT: DW_AT_reference  DW_FORM_flag_present
; CHECK: DW_TAG_subroutine_type     DW_CHILDREN_yes
; CHECK-NEXT: DW_AT_rvalue_reference DW_FORM_flag_present
;
; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG_subprogram
; CHECK:   DW_AT_name {{.*}}"l"
; CHECK-NOT: DW_TAG_subprogram
; CHECK:   DW_AT_reference [DW_FORM_flag_present] (true)

; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG_subprogram
; CHECK:   DW_AT_name {{.*}}"r"
; CHECK-NOT: DW_TAG_subprogram
; CHECK:   DW_AT_rvalue_reference [DW_FORM_flag_present] (true)

%class.A = type { i8 }

; Function Attrs: nounwind
define void @_Z1gv() #0 !dbg !17 {
  %a = alloca %class.A, align 1
  %pl = alloca { i64, i64 }, align 8
  %pr = alloca { i64, i64 }, align 8
  call void @llvm.dbg.declare(metadata %class.A* %a, metadata !24, metadata !DIExpression()), !dbg !25
  call void @llvm.dbg.declare(metadata { i64, i64 }* %pl, metadata !26, metadata !DIExpression()), !dbg !31
  store { i64, i64 } { i64 ptrtoint (void (%class.A*)* @_ZNKR1A1lEv to i64), i64 0 }, { i64, i64 }* %pl, align 8, !dbg !31
  call void @llvm.dbg.declare(metadata { i64, i64 }* %pr, metadata !32, metadata !DIExpression()), !dbg !35
  store { i64, i64 } { i64 ptrtoint (void (%class.A*)* @_ZNKO1A1rEv to i64), i64 0 }, { i64, i64 }* %pr, align 8, !dbg !35
  ret void, !dbg !36
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare void @_ZNKR1A1lEv(%class.A*)

declare void @_ZNKO1A1rEv(%class.A*)

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!21, !22}
!llvm.ident = !{!23}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.5 ", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !3, globals: !2, imports: !2)
!1 = !DIFile(filename: "debug-info-qualifiers.cpp", directory: "")
!2 = !{}
!3 = !{!4}
!4 = !DICompositeType(tag: DW_TAG_class_type, name: "A", line: 2, size: 8, align: 8, file: !5, elements: !6, identifier: "_ZTS1A")
!5 = !DIFile(filename: "debug-info-qualifiers.cpp", directory: "")
!6 = !{!7, !13}
!7 = !DISubprogram(name: "l", linkageName: "_ZNKR1A1lEv", line: 5, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagPrototyped | DIFlagLValueReference, isOptimized: false, scopeLine: 5, file: !5, scope: !4, type: !8)
!8 = !DISubroutineType(flags: DIFlagLValueReference, types: !9)
!9 = !{null, !10}
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer, baseType: !11)
!11 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !4)
!13 = !DISubprogram(name: "r", linkageName: "_ZNKO1A1rEv", line: 7, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagObjectPointer | DIFlagRValueReference, isOptimized: false, scopeLine: 7, file: !5, scope: !4, type: !14)
!14 = !DISubroutineType(flags: DIFlagRValueReference, types: !9)
!17 = distinct !DISubprogram(name: "g", linkageName: "_Z1gv", line: 10, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 10, file: !5, scope: !18, type: !19, retainedNodes: !2)
!18 = !DIFile(filename: "debug-info-qualifiers.cpp", directory: "")
!19 = !DISubroutineType(types: !20)
!20 = !{null}
!21 = !{i32 2, !"Dwarf Version", i32 4}
!22 = !{i32 1, !"Debug Info Version", i32 3}
!23 = !{!"clang version 3.5 "}
!24 = !DILocalVariable(name: "a", line: 11, scope: !17, file: !18, type: !4)
!25 = !DILocation(line: 11, scope: !17)
!26 = !DILocalVariable(name: "pl", line: 16, scope: !17, file: !18, type: !27)
!27 = !DIDerivedType(tag: DW_TAG_ptr_to_member_type, baseType: !28, extraData: !4)
!28 = !DISubroutineType(flags: DIFlagLValueReference, types: !29)
!29 = !{null, !30}
!30 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer, baseType: !4)
!31 = !DILocation(line: 16, scope: !17)
!32 = !DILocalVariable(name: "pr", line: 21, scope: !17, file: !18, type: !33)
!33 = !DIDerivedType(tag: DW_TAG_ptr_to_member_type, baseType: !34, extraData: !4)
!34 = !DISubroutineType(flags: DIFlagRValueReference, types: !29)
!35 = !DILocation(line: 21, scope: !17)
!36 = !DILocation(line: 22, scope: !17)
