; RUN: llc < %s -filetype=obj -o - | llvm-readobj - -codeview | FileCheck %s

; C++ source to regenerate:
; $ cat t.cpp
; struct A { int a; };
; struct B : virtual A { int b; };
; struct C : virtual A { int c; };
; struct D : B, C {
;   virtual void f(); // make vbptr not offset zero
;   int d;
; };
; D d;
; $ clang -fno-rtti -g -gcodeview t.cpp -emit-llvm -S -o t.ll -O1

; D's field list comes first.
; CHECK:        FieldList ({{.*}}) {
; CHECK-NEXT:     TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK-NEXT:     BaseClass {
; CHECK-NEXT:       TypeLeafKind: LF_BCLASS (0x1400)
; CHECK-NEXT:       AccessSpecifier: Public (0x3)
; CHECK-NEXT:       BaseType: B ({{.*}})
; CHECK-NEXT:       BaseOffset: 0x8
; CHECK-NEXT:     }
; CHECK-NEXT:     BaseClass {
; CHECK-NEXT:       TypeLeafKind: LF_BCLASS (0x1400)
; CHECK-NEXT:       AccessSpecifier: Public (0x3)
; CHECK-NEXT:       BaseType: C ({{.*}})
; CHECK-NEXT:       BaseOffset: 0x18
; CHECK-NEXT:     }
; CHECK-NEXT:     IndirectVirtualBaseClass {
; CHECK-NEXT:       TypeLeafKind: LF_IVBCLASS (0x1402)
; CHECK-NEXT:       AccessSpecifier: Public (0x3)
; CHECK-NEXT:       BaseType: A ({{.*}})
; CHECK-NEXT:       VBPtrType: const int* ({{.*}})
; CHECK-NEXT:       VBPtrOffset: 0x0
; CHECK-NEXT:       VBTableIndex: 0x1
; CHECK-NEXT:     }
; CHECK:        }

; Then B's field list.
; CHECK:        FieldList ({{.*}}) {
; CHECK-NEXT:     TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK-NEXT:     VirtualBaseClass {
; CHECK-NEXT:       TypeLeafKind: LF_VBCLASS (0x1401)
; CHECK-NEXT:       AccessSpecifier: Public (0x3)
; CHECK-NEXT:       BaseType: A ({{.*}})
; CHECK-NEXT:       VBPtrType: const int* ({{.*}})
; CHECK-NEXT:       VBPtrOffset: 0x0
; CHECK-NEXT:       VBTableIndex: 0x1
; CHECK-NEXT:     }
; CHECK:        }

; Then C's field list.
; CHECK:        FieldList ({{.*}}) {
; CHECK-NEXT:     TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK-NEXT:     VirtualBaseClass {
; CHECK-NEXT:       TypeLeafKind: LF_VBCLASS (0x1401)
; CHECK-NEXT:       AccessSpecifier: Public (0x3)
; CHECK-NEXT:       BaseType: A ({{.*}})
; CHECK-NEXT:       VBPtrType: const int* ({{.*}})
; CHECK-NEXT:       VBPtrOffset: 0x0
; CHECK-NEXT:       VBTableIndex: 0x1
; CHECK-NEXT:     }
; CHECK:        }

; ModuleID = 't.cpp'
source_filename = "t.cpp"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.0.23918"

%struct.D = type { i32 (...)**, %struct.B.base, %struct.C.base, i32, [4 x i8], %struct.A }
%struct.B.base = type { i32*, i32 }
%struct.C.base = type { i32*, i32 }
%struct.A = type { i32 }

$"\01??_8D@@7BB@@@" = comdat any

$"\01??_8D@@7BC@@@" = comdat any

$"\01??_7D@@6B@" = comdat any

@"\01?d@@3UD@@A" = local_unnamed_addr global %struct.D { i32 (...)** bitcast ([1 x i8*]* @"\01??_7D@@6B@" to i32 (...)**), %struct.B.base { i32* getelementptr inbounds ([2 x i32], [2 x i32]* @"\01??_8D@@7BB@@@", i32 0, i32 0), i32 0 }, %struct.C.base { i32* getelementptr inbounds ([2 x i32], [2 x i32]* @"\01??_8D@@7BC@@@", i32 0, i32 0), i32 0 }, i32 0, [4 x i8] zeroinitializer, %struct.A zeroinitializer }, align 8, !dbg !0
@"\01??_8D@@7BB@@@" = linkonce_odr unnamed_addr constant [2 x i32] [i32 0, i32 40], comdat
@"\01??_8D@@7BC@@@" = linkonce_odr unnamed_addr constant [2 x i32] [i32 0, i32 24], comdat
@"\01??_7D@@6B@" = linkonce_odr unnamed_addr constant [1 x i8*] [i8* bitcast (void (%struct.D*)* @"\01?f@D@@UEAAXXZ" to i8*)], comdat
@llvm.global_ctors = appending global [0 x { i32, void ()*, i8* }] zeroinitializer

declare void @"\01?f@D@@UEAAXXZ"(%struct.D*) unnamed_addr #0

attributes #0 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!1}
!llvm.module.flags = !{!31, !32, !33}
!llvm.ident = !{!34}

!0 = distinct !DIGlobalVariableExpression(var: !DIGlobalVariable(name: "d", linkageName: "\01?d@@3UD@@A", scope: !1, file: !5, line: 8, type: !6, isLocal: false, isDefinition: true))
!1 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !2, producer: "clang version 4.0.0 (http://llvm.org/git/clang.git 95626d54d6db7e13087089396a80ebaccc4ffe7c) (http://llvm.org/git/llvm.git 374b6e2fa0b230d13c0fb9ee7af69b2146bfad8a)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !3, globals: !4)
!2 = !DIFile(filename: "t.cpp", directory: "D:\5Csrc\5Cllvm\5Cbuild")
!3 = !{}
!4 = !{!0}
!5 = !DIFile(filename: "t.cpp", directory: "C:\5Cbuild\5Cllvm\5Cbuild")
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "D", file: !5, line: 4, size: 448, elements: !7, vtableHolder: !6, identifier: ".?AUD@@")
!7 = !{!8, !17, !22, !23, !24, !26, !27}
!8 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !6, baseType: !9, offset: 64)
!9 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "B", file: !5, line: 2, size: 192, elements: !10, vtableHolder: !9, identifier: ".?AUB@@")
!10 = !{!11, !16}
!11 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !9, baseType: !12, offset: 4, flags: DIFlagVirtual)
!12 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "A", file: !5, line: 1, size: 32, elements: !13, identifier: ".?AUA@@")
!13 = !{!14}
!14 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !12, file: !5, line: 1, baseType: !15, size: 32)
!15 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!16 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !9, file: !5, line: 2, baseType: !15, size: 32, offset: 64)
!17 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !6, baseType: !18, offset: 192)
!18 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "C", file: !5, line: 3, size: 192, elements: !19, vtableHolder: !18, identifier: ".?AUC@@")
!19 = !{!20, !21}
!20 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !18, baseType: !12, offset: 4, flags: DIFlagVirtual)
!21 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !18, file: !5, line: 3, baseType: !15, size: 32, offset: 64)
!22 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !6, baseType: !12, offset: 4, flags: DIFlagIndirectVirtualBase)
!23 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "__vtbl_ptr_type", baseType: null, size: 64)
!24 = !DIDerivedType(tag: DW_TAG_member, name: "_vptr$D", scope: !5, file: !5, baseType: !25, size: 64, flags: DIFlagArtificial)
!25 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !23, size: 64)
!26 = !DIDerivedType(tag: DW_TAG_member, name: "d", scope: !6, file: !5, line: 6, baseType: !15, size: 32, offset: 320)
!27 = !DISubprogram(name: "f", linkageName: "\01?f@D@@UEAAXXZ", scope: !6, file: !5, line: 5, type: !28, isLocal: false, isDefinition: false, scopeLine: 5, containingType: !6, virtuality: DW_VIRTUALITY_virtual, virtualIndex: 0, flags: DIFlagPrototyped | DIFlagIntroducedVirtual, isOptimized: true)
!28 = !DISubroutineType(types: !29)
!29 = !{null, !30}
!30 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!31 = !{i32 2, !"CodeView", i32 1}
!32 = !{i32 2, !"Debug Info Version", i32 3}
!33 = !{i32 1, !"PIC Level", i32 2}
!34 = !{!"clang version 4.0.0 (http://llvm.org/git/clang.git 95626d54d6db7e13087089396a80ebaccc4ffe7c) (http://llvm.org/git/llvm.git 374b6e2fa0b230d13c0fb9ee7af69b2146bfad8a)"}
