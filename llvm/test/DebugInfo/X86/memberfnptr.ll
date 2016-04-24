; struct A {
;   void foo();
; };
;  
; void (A::*p)() = &A::foo;
;
; RUN: llc -filetype=obj -o - %s | llvm-dwarfdump -debug-dump=info - | FileCheck %s
; Check that the member function pointer is emitted without a DW_AT_size attribute.
; CHECK: DW_TAG_ptr_to_member_type
; CHECK-NOT: DW_AT_{{.*}}size
; CHECK: DW_TAG
;
; ModuleID = 'memberfnptr.cpp'
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx"

%struct.A = type { i8 }

@p = global { i64, i64 } { i64 ptrtoint (void (%struct.A*)* @_ZN1A3fooEv to i64), i64 0 }, align 8

declare void @_ZN1A3fooEv(%struct.A*)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!14, !15, !16}
!llvm.ident = !{!17}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.6.0 ", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !3, globals: !10, imports: !2)
!1 = !DIFile(filename: "memberfnptr.cpp", directory: "")
!2 = !{}
!3 = !{!4}
!4 = !DICompositeType(tag: DW_TAG_structure_type, name: "A", line: 1, size: 8, align: 8, file: !1, elements: !5, identifier: "_ZTS1A")
!5 = !{!6}
!6 = !DISubprogram(name: "foo", linkageName: "_ZN1A3fooEv", line: 2, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 2, file: !1, scope: !4, type: !7)
!7 = !DISubroutineType(types: !8)
!8 = !{null, !9}
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer, baseType: !4)
!10 = !{!11}
!11 = !DIGlobalVariable(name: "p", line: 5, isLocal: false, isDefinition: true, scope: null, file: !12, type: !13, variable: { i64, i64 }* @p)
!12 = !DIFile(filename: "memberfnptr.cpp", directory: "")
!13 = !DIDerivedType(tag: DW_TAG_ptr_to_member_type, size: 64, baseType: !7, extraData: !4)
!14 = !{i32 2, !"Dwarf Version", i32 2}
!15 = !{i32 2, !"Debug Info Version", i32 3}
!16 = !{i32 1, !"PIC Level", i32 2}
!17 = !{!"clang version 3.6.0 "}
