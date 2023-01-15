; RUN: llc < %s -filetype=obj -o - | llvm-readobj - --codeview | FileCheck %s
; RUN: llc < %s -o - | llvm-mc -filetype=obj --triple=x86_64-windows | llvm-readobj - --codeview | FileCheck %s

; C++ source to regenerate:
; $ cat t.cpp
; struct A { int a; };
; struct B : virtual A { int b; virtual int get() { return b; } };
; struct C : virtual A { int c; virtual int get() { return c; } };
; struct D : B, C {
;   virtual void f(); // make vbptr not offset zero
;   int d;
; };
; D d;
; $ clang -fno-rtti -g -gcodeview t.cpp -emit-llvm -S -o t.ll -O1

; struct B's field list comes first.
; CHECK:        FieldList ({{.*}}) {
; CHECK-NEXT:     TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK-NEXT:     VirtualBaseClass {
; CHECK-NEXT:       TypeLeafKind: LF_VBCLASS (0x1401)
; CHECK-NEXT:       AccessSpecifier: Public (0x3)
; CHECK-NEXT:       BaseType: A ({{.*}})
; CHECK-NEXT:       VBPtrType: const int* ({{.*}})
; CHECK-NEXT:       VBPtrOffset: 0x8
; CHECK-NEXT:       VBTableIndex: 0x1
; CHECK-NEXT:     }
; CHECK:        }

; struct A's field list comes next.
; CHECK:       FieldList ({{.*}})
; CHECK-NEXT:    TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK:       }

; struct C's field list comes next.
; CHECK:       FieldList ({{.*}})
; CHECK-NEXT:    TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK-NEXT:    VirtualBaseClass {
; CHECK-NEXT:      TypeLeafKind: LF_VBCLASS (0x1401)
; CHECK-NEXT:      AccessSpecifier: Public (0x3)
; CHECK-NEXT:      BaseType: A ({{.*}})
; CHECK-NEXT:      VBPtrType: const int* ({{.*}})
; CHECK-NEXT:      VBPtrOffset: 0x8
; CHECK-NEXT:      VBTableIndex: 0x1
; CHECK-NEXT:    }
; CHECK:       }

; struct D's field list is last.
; CHECK:       FieldList ({{.*}}) {
; CHECK-NEXT:    TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK-NEXT:    BaseClass {
; CHECK-NEXT:      TypeLeafKind: LF_BCLASS (0x1400)
; CHECK-NEXT:      AccessSpecifier: Public (0x3)
; CHECK-NEXT:      BaseType: B ({{.*}})
; CHECK-NEXT:      BaseOffset: 0x0
; CHECK-NEXT:    }
; CHECK-NEXT:    BaseClass {
; CHECK-NEXT:      TypeLeafKind: LF_BCLASS (0x1400)
; CHECK-NEXT:      AccessSpecifier: Public (0x3)
; CHECK-NEXT:      BaseType: C ({{.*}})
; CHECK-NEXT:      BaseOffset: 0x18
; CHECK-NEXT:    }
; CHECK-NEXT:    IndirectVirtualBaseClass {
; CHECK-NEXT:      TypeLeafKind: LF_IVBCLASS (0x1402)
; CHECK-NEXT:      AccessSpecifier: Public (0x3)
; CHECK-NEXT:      BaseType: A ({{.*}})
; CHECK-NEXT:      VBPtrType: const int* ({{.*}})
; CHECK-NEXT:      VBPtrOffset: 0x8
; CHECK-NEXT:      VBTableIndex: 0x1
; CHECK-NEXT:    }
; CHECK:       }

; ModuleID = 't.cpp'
source_filename = "t.cpp"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.12.25835"

%struct.D = type { %struct.B.base, %struct.C.base, i32, [4 x i8], %struct.A }
%struct.B.base = type { ptr, ptr, i32 }
%struct.C.base = type { ptr, ptr, i32 }
%struct.A = type { i32 }
%struct.B = type { ptr, ptr, i32, [4 x i8], %struct.A }
%struct.C = type { ptr, ptr, i32, [4 x i8], %struct.A }

$"?get@B@@UEAAHXZ" = comdat any

$"?get@C@@UEAAHXZ" = comdat any

$"??_8D@@7BB@@@" = comdat any

$"??_8D@@7BC@@@" = comdat any

$"??_7D@@6BB@@@" = comdat any

$"??_7D@@6BC@@@" = comdat any

@"?d@@3UD@@A" = dso_local local_unnamed_addr global %struct.D { %struct.B.base { ptr @"??_7D@@6BB@@@", ptr @"??_8D@@7BB@@@", i32 0 }, %struct.C.base { ptr @"??_7D@@6BC@@@", ptr @"??_8D@@7BC@@@", i32 0 }, i32 0, [4 x i8] zeroinitializer, %struct.A zeroinitializer }, align 8, !dbg !0
@"??_8D@@7BB@@@" = linkonce_odr unnamed_addr constant [2 x i32] [i32 -8, i32 48], comdat
@"??_8D@@7BC@@@" = linkonce_odr unnamed_addr constant [2 x i32] [i32 -8, i32 24], comdat
@"??_7D@@6BB@@@" = linkonce_odr unnamed_addr constant { [2 x ptr] } { [2 x ptr] [ptr @"?get@B@@UEAAHXZ", ptr @"?f@D@@UEAAXXZ"] }, comdat
@"??_7D@@6BC@@@" = linkonce_odr unnamed_addr constant { [1 x ptr] } { [1 x ptr] [ptr @"?get@C@@UEAAHXZ"] }, comdat
@llvm.global_ctors = appending global [0 x { i32, ptr, ptr }] zeroinitializer

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local i32 @"?get@B@@UEAAHXZ"(ptr %this) unnamed_addr #0 comdat align 2 !dbg !46 {
entry:
  call void @llvm.dbg.value(metadata ptr %this, metadata !48, metadata !DIExpression()), !dbg !50
  %b = getelementptr inbounds %struct.B, ptr %this, i64 0, i32 2, !dbg !51
  %0 = load i32, ptr %b, align 8, !dbg !51, !tbaa !52
  ret i32 %0, !dbg !51
}

declare dso_local void @"?f@D@@UEAAXXZ"(ptr) unnamed_addr #1

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local i32 @"?get@C@@UEAAHXZ"(ptr %this) unnamed_addr #0 comdat align 2 !dbg !57 {
entry:
  call void @llvm.dbg.value(metadata ptr %this, metadata !59, metadata !DIExpression()), !dbg !61
  %c = getelementptr inbounds %struct.C, ptr %this, i64 0, i32 2, !dbg !62
  %0 = load i32, ptr %c, align 8, !dbg !62, !tbaa !63
  ret i32 %0, !dbg !62
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!41, !42, !43, !44}
!llvm.ident = !{!45}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "d", linkageName: "?d@@3UD@@A", scope: !2, file: !3, line: 8, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 7.0.0 (trunk)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "t.cpp", directory: "C:\5CPath\5CTo\5CDirectory", checksumkind: CSK_MD5, checksum: "7477d4db6bf8a461a719bcaab9c6d65e")
!4 = !{}
!5 = !{!0}
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "D", file: !3, line: 4, size: 512, flags: DIFlagTypePassByReference, elements: !7, vtableHolder: !9, identifier: ".?AUD@@")
!7 = !{!8, !24, !34, !35, !36, !37}
!8 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !6, baseType: !9, extraData: i32 0)
!9 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "B", file: !3, line: 2, size: 256, flags: DIFlagTypePassByReference, elements: !10, vtableHolder: !9, identifier: ".?AUB@@")
!10 = !{!11, !16, !17, !19, !20}
!11 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !9, baseType: !12, offset: 4, flags: DIFlagVirtual, extraData: i32 8)
!12 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "A", file: !3, line: 1, size: 32, flags: DIFlagTypePassByValue, elements: !13, identifier: ".?AUA@@")
!13 = !{!14}
!14 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !12, file: !3, line: 1, baseType: !15, size: 32)
!15 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!16 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "__vtbl_ptr_type", baseType: null, size: 64)
!17 = !DIDerivedType(tag: DW_TAG_member, name: "_vptr$B", scope: !3, file: !3, baseType: !18, size: 64, flags: DIFlagArtificial)
!18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !16, size: 64)
!19 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !9, file: !3, line: 2, baseType: !15, size: 32, offset: 128)
!20 = !DISubprogram(name: "get", linkageName: "?get@B@@UEAAHXZ", scope: !9, file: !3, line: 2, type: !21, isLocal: false, isDefinition: false, scopeLine: 2, containingType: !9, virtuality: DW_VIRTUALITY_virtual, virtualIndex: 0, flags: DIFlagPrototyped | DIFlagIntroducedVirtual, isOptimized: true)
!21 = !DISubroutineType(types: !22)
!22 = !{!15, !23}
!23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!24 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !6, baseType: !25, offset: 192, extraData: i32 0)
!25 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "C", file: !3, line: 3, size: 256, flags: DIFlagTypePassByReference, elements: !26, vtableHolder: !25, identifier: ".?AUC@@")
!26 = !{!27, !16, !28, !29, !30}
!27 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !25, baseType: !12, offset: 4, flags: DIFlagVirtual, extraData: i32 8)
!28 = !DIDerivedType(tag: DW_TAG_member, name: "_vptr$C", scope: !3, file: !3, baseType: !18, size: 64, flags: DIFlagArtificial)
!29 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !25, file: !3, line: 3, baseType: !15, size: 32, offset: 128)
!30 = !DISubprogram(name: "get", linkageName: "?get@C@@UEAAHXZ", scope: !25, file: !3, line: 3, type: !31, isLocal: false, isDefinition: false, scopeLine: 3, containingType: !25, virtuality: DW_VIRTUALITY_virtual, virtualIndex: 0, flags: DIFlagPrototyped | DIFlagIntroducedVirtual, isOptimized: true)
!31 = !DISubroutineType(types: !32)
!32 = !{!15, !33}
!33 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !25, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!34 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !6, baseType: !12, offset: 4, flags: DIFlagIndirectVirtualBase, extraData: i32 8)
!35 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "__vtbl_ptr_type", baseType: null, size: 128)
!36 = !DIDerivedType(tag: DW_TAG_member, name: "d", scope: !6, file: !3, line: 6, baseType: !15, size: 32, offset: 384)
!37 = !DISubprogram(name: "f", linkageName: "?f@D@@UEAAXXZ", scope: !6, file: !3, line: 5, type: !38, isLocal: false, isDefinition: false, scopeLine: 5, containingType: !6, virtuality: DW_VIRTUALITY_virtual, virtualIndex: 1, flags: DIFlagPrototyped | DIFlagIntroducedVirtual, isOptimized: true)
!38 = !DISubroutineType(types: !39)
!39 = !{null, !40}
!40 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!41 = !{i32 2, !"CodeView", i32 1}
!42 = !{i32 2, !"Debug Info Version", i32 3}
!43 = !{i32 1, !"wchar_size", i32 2}
!44 = !{i32 7, !"PIC Level", i32 2}
!45 = !{!"clang version 7.0.0 (trunk)"}
!46 = distinct !DISubprogram(name: "get", linkageName: "?get@B@@UEAAHXZ", scope: !9, file: !3, line: 2, type: !21, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: true, unit: !2, declaration: !20, retainedNodes: !47)
!47 = !{!48}
!48 = !DILocalVariable(name: "this", arg: 1, scope: !46, type: !49, flags: DIFlagArtificial | DIFlagObjectPointer)
!49 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 64)
!50 = !DILocation(line: 0, scope: !46)
!51 = !DILocation(line: 2, scope: !46)
!52 = !{!53, !54, i64 16}
!53 = !{!"?AUB@@", !54, i64 16}
!54 = !{!"int", !55, i64 0}
!55 = !{!"omnipotent char", !56, i64 0}
!56 = !{!"Simple C++ TBAA"}
!57 = distinct !DISubprogram(name: "get", linkageName: "?get@C@@UEAAHXZ", scope: !25, file: !3, line: 3, type: !31, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: true, unit: !2, declaration: !30, retainedNodes: !58)
!58 = !{!59}
!59 = !DILocalVariable(name: "this", arg: 1, scope: !57, type: !60, flags: DIFlagArtificial | DIFlagObjectPointer)
!60 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !25, size: 64)
!61 = !DILocation(line: 0, scope: !57)
!62 = !DILocation(line: 3, scope: !57)
!63 = !{!64, !54, i64 16}
!64 = !{!"?AUC@@", !54, i64 16}
