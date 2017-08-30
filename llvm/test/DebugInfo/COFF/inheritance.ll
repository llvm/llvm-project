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

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!32, !33, !34}
!llvm.ident = !{!35}

!0 = distinct !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "d", linkageName: "\01?d@@3UD@@A", scope: !2, file: !6, line: 8, type: !7, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 4.0.0 (http://llvm.org/git/clang.git 95626d54d6db7e13087089396a80ebaccc4ffe7c) (http://llvm.org/git/llvm.git 374b6e2fa0b230d13c0fb9ee7af69b2146bfad8a)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "t.cpp", directory: "D:\5Csrc\5Cllvm\5Cbuild")
!4 = !{}
!5 = !{!0}
!6 = !DIFile(filename: "t.cpp", directory: "C:\5Cbuild\5Cllvm\5Cbuild")
!7 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "D", file: !6, line: 4, size: 448, elements: !8, vtableHolder: !7, identifier: ".?AUD@@")
!8 = !{!9, !18, !23, !24, !25, !27, !28}
!9 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !7, baseType: !10, offset: 64)
!10 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "B", file: !6, line: 2, size: 192, elements: !11, vtableHolder: !10, identifier: ".?AUB@@")
!11 = !{!12, !17}
!12 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !10, baseType: !13, offset: 4, flags: DIFlagVirtual)
!13 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "A", file: !6, line: 1, size: 32, elements: !14, identifier: ".?AUA@@")
!14 = !{!15}
!15 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !13, file: !6, line: 1, baseType: !16, size: 32)
!16 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!17 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !10, file: !6, line: 2, baseType: !16, size: 32, offset: 64)
!18 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !7, baseType: !19, offset: 192)
!19 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "C", file: !6, line: 3, size: 192, elements: !20, vtableHolder: !19, identifier: ".?AUC@@")
!20 = !{!21, !22}
!21 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !19, baseType: !13, offset: 4, flags: DIFlagVirtual)
!22 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !19, file: !6, line: 3, baseType: !16, size: 32, offset: 64)
!23 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !7, baseType: !13, offset: 4, flags: DIFlagIndirectVirtualBase)
!24 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "__vtbl_ptr_type", baseType: null, size: 64)
!25 = !DIDerivedType(tag: DW_TAG_member, name: "_vptr$D", scope: !6, file: !6, baseType: !26, size: 64, flags: DIFlagArtificial)
!26 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !24, size: 64)
!27 = !DIDerivedType(tag: DW_TAG_member, name: "d", scope: !7, file: !6, line: 6, baseType: !16, size: 32, offset: 320)
!28 = !DISubprogram(name: "f", linkageName: "\01?f@D@@UEAAXXZ", scope: !7, file: !6, line: 5, type: !29, isLocal: false, isDefinition: false, scopeLine: 5, containingType: !7, virtuality: DW_VIRTUALITY_virtual, virtualIndex: 0, flags: DIFlagPrototyped | DIFlagIntroducedVirtual, isOptimized: true)
!29 = !DISubroutineType(types: !30)
!30 = !{null, !31}
!31 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!32 = !{i32 2, !"CodeView", i32 1}
!33 = !{i32 2, !"Debug Info Version", i32 3}
!34 = !{i32 1, !"PIC Level", i32 2}
!35 = !{!"clang version 4.0.0 (http://llvm.org/git/clang.git 95626d54d6db7e13087089396a80ebaccc4ffe7c) (http://llvm.org/git/llvm.git 374b6e2fa0b230d13c0fb9ee7af69b2146bfad8a)"}

