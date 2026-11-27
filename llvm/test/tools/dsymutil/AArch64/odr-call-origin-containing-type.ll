; A call site names the declaration of its callee, and a class names its vtable
; holder. Both are in the type unit, so the compile unit gets no copy of the
; class around them. A callee defined in the compile unit has an address, and
; the call site keeps naming that definition.
;
; struct S { void ext(); };
; struct Base { virtual void f(); int x; };
; namespace { struct Derived : Base {}; }
; void sink();
; void local() {}
; void call(S *s) {
;   Derived d;
;   s->ext();
;   sink();
;   local();
; }

; RUN: rm -rf %t && split-file %s %t
; RUN: %llc_dwarf -mtriple=arm64-apple-darwin -filetype=obj %t/input.ll -o %t/input.o

; RUN: dsymutil -f --linker=parallel -oso-prepend-path=%t -y %t/input.map -o %t/parallel.dwarf
; RUN: llvm-dwarfdump --verify %t/parallel.dwarf
; RUN: llvm-dwarfdump --debug-info %t/parallel.dwarf | FileCheck %s \
; RUN:   --implicit-check-not=DW_TAG_structure_type

; CHECK: DW_AT_name ("__artificial_type_unit")
; CHECK: 0x[[BASE:[0-9a-f]+]]: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_containing_type (0x[[BASE]] "Base")
; CHECK: DW_TAG_structure_type
; CHECK: DW_AT_name ("S")
; CHECK: 0x[[EXT:[0-9a-f]+]]: DW_TAG_subprogram
; CHECK: DW_AT_name ("ext")
; CHECK: 0x[[SINK:[0-9a-f]+]]: DW_TAG_subprogram
; CHECK: DW_AT_name ("sink")

; CHECK: DW_AT_name ("a.cpp")
; CHECK: 0x[[LOCAL:[0-9a-f]+]]: DW_TAG_subprogram
; CHECK: DW_AT_name ("local")
; CHECK: DW_AT_call_origin (0x00000000[[EXT]] "_ZN1S3extEv")
; CHECK: DW_AT_call_origin (0x00000000[[SINK]] "_Z4sinkv")
; CHECK: DW_AT_call_origin (0x[[LOCAL]] "_Z5localv")
; CHECK: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_containing_type (0x00000000[[BASE]] "Base")
; CHECK: DW_AT_name ("Derived")

;--- input.ll
target triple = "arm64-apple-darwin"

declare !dbg !8 void @_ZN1S3extEv(ptr)

declare !dbg !20 void @_Z4sinkv()

define void @_Z5localv() !dbg !21 {
  ret void, !dbg !22
}

define void @_Z4callP1S(ptr %s) !dbg !23 {
  call void @_ZN1S3extEv(ptr %s), !dbg !30
  call void @_Z4sinkv(), !dbg !30
  call void @_Z5localv(), !dbg !30
  ret void, !dbg !30
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "a.cpp", directory: "/tmp")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DISubroutineType(types: !5)
!5 = !{null}
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !1, line: 1, size: 8, flags: DIFlagTypePassByValue, elements: !7, identifier: "_ZTS1S")
!7 = !{!8}
!8 = !DISubprogram(name: "ext", linkageName: "_ZN1S3extEv", scope: !6, file: !1, line: 1, type: !9, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!9 = !DISubroutineType(types: !10)
!10 = !{null, !11}
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!12 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Base", file: !1, line: 2, size: 64, flags: DIFlagTypePassByReference | DIFlagNonTrivial, elements: !13, vtableHolder: !12, identifier: "_ZTS4Base")
!13 = !{!14}
!14 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !12, file: !1, line: 2, baseType: !15, size: 32)
!15 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!16 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Derived", scope: !17, file: !1, line: 3, size: 64, flags: DIFlagTypePassByReference | DIFlagNonTrivial, elements: !18, vtableHolder: !12)
!17 = !DINamespace(scope: null)
!18 = !{!19}
!19 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !16, baseType: !12, extraData: i32 0)
!20 = !DISubprogram(name: "sink", linkageName: "_Z4sinkv", scope: !1, file: !1, line: 4, type: !4, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!21 = distinct !DISubprogram(name: "local", linkageName: "_Z5localv", scope: !1, file: !1, line: 5, type: !4, scopeLine: 5, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!22 = !DILocation(line: 5, scope: !21)
!23 = distinct !DISubprogram(name: "call", linkageName: "_Z4callP1S", scope: !1, file: !1, line: 6, type: !24, scopeLine: 6, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !27)
!24 = !DISubroutineType(types: !25)
!25 = !{null, !26}
!26 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 64)
!27 = !{!28, !29}
!28 = !DILocalVariable(name: "s", arg: 1, scope: !23, file: !1, line: 6, type: !26)
!29 = !DILocalVariable(name: "d", scope: !23, file: !1, line: 7, type: !16)
!30 = !DILocation(line: 8, scope: !23)

;--- input.map
---
triple: 'arm64-apple-darwin'
objects:
  - filename: input.o
    symbols:
      - { sym: __Z5localv, objAddr: 0x0, binAddr: 0x1000, size: 0x4 }
      - { sym: __Z4callP1S, objAddr: 0x4, binAddr: 0x1004, size: 0x18 }
...
