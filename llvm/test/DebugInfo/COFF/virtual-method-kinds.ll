; RUN: llc < %s -filetype=obj | llvm-readobj - --codeview | FileCheck %s
; RUN: llc < %s | llvm-mc -filetype=obj --triple=x86_64-windows | llvm-readobj - --codeview | FileCheck %s

; Check for the appropriate MethodKind below.

; C++ source used to generate IR:
; $ cat t.cpp
; struct A {
;   virtual void f();	  // IntroducingVirtual
;   virtual void g() = 0; // PureIntroducingVirtual
; };
; struct B : A {
;   void f() override = 0; // PureVirtual
;   void g() override;	   // Virtual
; };
; struct C : B {
;   void f() override;     // Virtual
;   void g() override;     // Virtual
; };
; C *p = new C;
; $ clang t.cpp -S -emit-llvm -g -gcodeview -o t.ll

; CHECK:      OneMethod {
; CHECK-NEXT:   TypeLeafKind: LF_ONEMETHOD (0x1511)
; CHECK-NEXT:   AccessSpecifier: Public (0x3)
; CHECK-NEXT:   MethodKind: Virtual (0x1)
; CHECK-NEXT:   Type: void C::() ({{.*}})
; CHECK-NEXT:   Name: f
; CHECK-NEXT: }
; CHECK-NEXT: OneMethod {
; CHECK-NEXT:   TypeLeafKind: LF_ONEMETHOD (0x1511)
; CHECK-NEXT:   AccessSpecifier: Public (0x3)
; CHECK-NEXT:   MethodKind: Virtual (0x1)
; CHECK-NEXT:   Type: void C::() ({{.*}})
; CHECK-NEXT:   Name: g
; CHECK-NEXT: }

; CHECK:      OneMethod {
; CHECK-NEXT:   TypeLeafKind: LF_ONEMETHOD (0x1511)
; CHECK-NEXT:   AccessSpecifier: Public (0x3)
; CHECK-NEXT:   MethodKind: PureVirtual (0x5)
; CHECK-NEXT:   Type: void B::() ({{.*}})
; CHECK-NEXT:   Name: f
; CHECK-NEXT: }
; CHECK-NEXT: OneMethod {
; CHECK-NEXT:   TypeLeafKind: LF_ONEMETHOD (0x1511)
; CHECK-NEXT:   AccessSpecifier: Public (0x3)
; CHECK-NEXT:   MethodKind: Virtual (0x1)
; CHECK-NEXT:   Type: void B::() ({{.*}})
; CHECK-NEXT:   Name: g
; CHECK-NEXT: }

; CHECK:      OneMethod {
; CHECK-NEXT:   TypeLeafKind: LF_ONEMETHOD (0x1511)
; CHECK-NEXT:   AccessSpecifier: Public (0x3)
; CHECK-NEXT:   MethodKind: IntroducingVirtual (0x4)
; CHECK-NEXT:   Type: void A::() ({{.*}})
; CHECK-NEXT:   VFTableOffset: 0x0
; CHECK-NEXT:   Name: f
; CHECK-NEXT: }
; CHECK-NEXT: OneMethod {
; CHECK-NEXT:   TypeLeafKind: LF_ONEMETHOD (0x1511)
; CHECK-NEXT:   AccessSpecifier: Public (0x3)
; CHECK-NEXT:   MethodKind: PureIntroducingVirtual (0x6)
; CHECK-NEXT:   Type: void A::() ({{.*}})
; CHECK-NEXT:   VFTableOffset: 0x8
; CHECK-NEXT:   Name: g
; CHECK-NEXT: }

; ModuleID = 't.cpp'
source_filename = "t.cpp"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.0.23918"

%struct.C = type { %struct.B }
%struct.B = type { %struct.A }
%struct.A = type { ptr }
%rtti.CompleteObjectLocator = type { i32, i32, i32, i32, i32, i32 }
%rtti.TypeDescriptor7 = type { ptr, ptr, [8 x i8] }
%rtti.ClassHierarchyDescriptor = type { i32, i32, i32, i32 }
%rtti.BaseClassDescriptor = type { i32, i32, i32, i32, i32, i32, i32 }

$"\01??0C@@QEAA@XZ" = comdat any

$"\01??0B@@QEAA@XZ" = comdat any

$"\01??0A@@QEAA@XZ" = comdat any

$"\01??_7C@@6B@" = comdat largest

$"\01??_R4C@@6B@" = comdat any

$"\01??_R0?AUC@@@8" = comdat any

$"\01??_R3C@@8" = comdat any

$"\01??_R2C@@8" = comdat any

$"\01??_R1A@?0A@EA@C@@8" = comdat any

$"\01??_R1A@?0A@EA@B@@8" = comdat any

$"\01??_R0?AUB@@@8" = comdat any

$"\01??_R3B@@8" = comdat any

$"\01??_R2B@@8" = comdat any

$"\01??_R1A@?0A@EA@A@@8" = comdat any

$"\01??_R0?AUA@@@8" = comdat any

$"\01??_R3A@@8" = comdat any

$"\01??_R2A@@8" = comdat any

$"\01??_7B@@6B@" = comdat largest

$"\01??_R4B@@6B@" = comdat any

$"\01??_7A@@6B@" = comdat largest

$"\01??_R4A@@6B@" = comdat any

@"\01?p@@3PEAUC@@EA" = global ptr null, align 8, !dbg !0
@0 = private unnamed_addr constant [3 x ptr] [ptr @"\01??_R4C@@6B@", ptr @"\01?f@C@@UEAAXXZ", ptr @"\01?g@C@@UEAAXXZ"], comdat($"\01??_7C@@6B@")
@"\01??_R4C@@6B@" = linkonce_odr constant %rtti.CompleteObjectLocator { i32 1, i32 0, i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"\01??_R0?AUC@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"\01??_R3C@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"\01??_R4C@@6B@" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"\01??_7type_info@@6B@" = external constant ptr
@"\01??_R0?AUC@@@8" = linkonce_odr global %rtti.TypeDescriptor7 { ptr @"\01??_7type_info@@6B@", ptr null, [8 x i8] c".?AUC@@\00" }, comdat
@__ImageBase = external constant i8
@"\01??_R3C@@8" = linkonce_odr constant %rtti.ClassHierarchyDescriptor { i32 0, i32 0, i32 3, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"\01??_R2C@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"\01??_R2C@@8" = linkonce_odr constant [4 x i32] [i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"\01??_R1A@?0A@EA@C@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"\01??_R1A@?0A@EA@B@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"\01??_R1A@?0A@EA@A@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0], comdat
@"\01??_R1A@?0A@EA@C@@8" = linkonce_odr constant %rtti.BaseClassDescriptor { i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"\01??_R0?AUC@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 2, i32 0, i32 -1, i32 0, i32 64, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"\01??_R3C@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"\01??_R1A@?0A@EA@B@@8" = linkonce_odr constant %rtti.BaseClassDescriptor { i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"\01??_R0?AUB@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 1, i32 0, i32 -1, i32 0, i32 64, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"\01??_R3B@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"\01??_R0?AUB@@@8" = linkonce_odr global %rtti.TypeDescriptor7 { ptr @"\01??_7type_info@@6B@", ptr null, [8 x i8] c".?AUB@@\00" }, comdat
@"\01??_R3B@@8" = linkonce_odr constant %rtti.ClassHierarchyDescriptor { i32 0, i32 0, i32 2, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"\01??_R2B@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"\01??_R2B@@8" = linkonce_odr constant [3 x i32] [i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"\01??_R1A@?0A@EA@B@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"\01??_R1A@?0A@EA@A@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0], comdat
@"\01??_R1A@?0A@EA@A@@8" = linkonce_odr constant %rtti.BaseClassDescriptor { i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"\01??_R0?AUA@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0, i32 0, i32 -1, i32 0, i32 64, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"\01??_R3A@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"\01??_R0?AUA@@@8" = linkonce_odr global %rtti.TypeDescriptor7 { ptr @"\01??_7type_info@@6B@", ptr null, [8 x i8] c".?AUA@@\00" }, comdat
@"\01??_R3A@@8" = linkonce_odr constant %rtti.ClassHierarchyDescriptor { i32 0, i32 0, i32 1, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"\01??_R2A@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"\01??_R2A@@8" = linkonce_odr constant [2 x i32] [i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"\01??_R1A@?0A@EA@A@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0], comdat
@1 = private unnamed_addr constant [3 x ptr] [ptr @"\01??_R4B@@6B@", ptr @_purecall, ptr @"\01?g@B@@UEAAXXZ"], comdat($"\01??_7B@@6B@")
@"\01??_R4B@@6B@" = linkonce_odr constant %rtti.CompleteObjectLocator { i32 1, i32 0, i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"\01??_R0?AUB@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"\01??_R3B@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"\01??_R4B@@6B@" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@2 = private unnamed_addr constant [3 x ptr] [ptr @"\01??_R4A@@6B@", ptr @"\01?f@A@@UEAAXXZ", ptr @_purecall], comdat($"\01??_7A@@6B@")
@"\01??_R4A@@6B@" = linkonce_odr constant %rtti.CompleteObjectLocator { i32 1, i32 0, i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"\01??_R0?AUA@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"\01??_R3A@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"\01??_R4A@@6B@" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_t.cpp, ptr null }]

@"\01??_7C@@6B@" = unnamed_addr alias ptr, getelementptr inbounds ([3 x ptr], ptr @0, i32 0, i32 1)
@"\01??_7B@@6B@" = unnamed_addr alias ptr, getelementptr inbounds ([3 x ptr], ptr @1, i32 0, i32 1)
@"\01??_7A@@6B@" = unnamed_addr alias ptr, getelementptr inbounds ([3 x ptr], ptr @2, i32 0, i32 1)

; Function Attrs: uwtable
define internal void @"\01??__Ep@@YAXXZ"() #0 !dbg !40 {
entry:
  %call = call ptr @"\01??2@YAPEAX_K@Z"(i64 8) #5, !dbg !43
  %call1 = call ptr @"\01??0C@@QEAA@XZ"(ptr %call) #6, !dbg !44
  store ptr %call, ptr @"\01?p@@3PEAUC@@EA", align 8, !dbg !43
  ret void, !dbg !44
}

; Function Attrs: nobuiltin
declare noalias ptr @"\01??2@YAPEAX_K@Z"(i64) #1

; Function Attrs: inlinehint nounwind uwtable

define linkonce_odr ptr @"\01??0C@@QEAA@XZ"(ptr returned %this) unnamed_addr #2 comdat align 2 !dbg !45 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  call void @llvm.dbg.declare(metadata ptr %this.addr, metadata !47, metadata !48), !dbg !49
  %this1 = load ptr, ptr %this.addr, align 8
  %call = call ptr @"\01??0B@@QEAA@XZ"(ptr %this1) #6, !dbg !50
  store ptr @"\01??_7C@@6B@", ptr %this1, align 8, !dbg !50
  ret ptr %this1, !dbg !50
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #3

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr ptr @"\01??0B@@QEAA@XZ"(ptr returned %this) unnamed_addr #2 comdat align 2 !dbg !51 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  call void @llvm.dbg.declare(metadata ptr %this.addr, metadata !53, metadata !48), !dbg !55
  %this1 = load ptr, ptr %this.addr, align 8
  %call = call ptr @"\01??0A@@QEAA@XZ"(ptr %this1) #6, !dbg !56
  store ptr @"\01??_7B@@6B@", ptr %this1, align 8, !dbg !56
  ret ptr %this1, !dbg !56
}

declare void @"\01?f@C@@UEAAXXZ"(ptr) unnamed_addr #4

declare void @"\01?g@C@@UEAAXXZ"(ptr) unnamed_addr #4

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr ptr @"\01??0A@@QEAA@XZ"(ptr returned %this) unnamed_addr #2 comdat align 2 !dbg !57 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  call void @llvm.dbg.declare(metadata ptr %this.addr, metadata !59, metadata !48), !dbg !61
  %this1 = load ptr, ptr %this.addr, align 8
  store ptr @"\01??_7A@@6B@", ptr %this1, align 8, !dbg !62
  ret ptr %this1, !dbg !62
}

declare void @_purecall() unnamed_addr

declare void @"\01?g@B@@UEAAXXZ"(ptr) unnamed_addr #4

declare void @"\01?f@A@@UEAAXXZ"(ptr) unnamed_addr #4

; Function Attrs: uwtable
define internal void @_GLOBAL__sub_I_t.cpp() #0 !dbg !63 {
entry:
  call void @"\01??__Ep@@YAXXZ"(), !dbg !65
  ret void
}

attributes #0 = { uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nobuiltin "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { inlinehint nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind readnone }
attributes #4 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { builtin }
attributes #6 = { nounwind }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!36, !37, !38}
!llvm.ident = !{!39}

!0 = distinct !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "p", linkageName: "\01?p@@3PEAUC@@EA", scope: !2, file: !3, line: 13, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 3.9.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "t.cpp", directory: "D:\5Csrc\5Cllvm\5Cbuild")
!4 = !{}
!5 = !{!0}
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64, align: 64)
!7 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "C", file: !3, line: 9, size: 64, align: 64, elements: !8, vtableHolder: !13, identifier: ".?AUC@@")
!8 = !{!9, !31, !35}
!9 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !7, baseType: !10)
!10 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "B", file: !3, line: 5, size: 64, align: 64, elements: !11, vtableHolder: !13, identifier: ".?AUB@@")
!11 = !{!12, !26, !30}
!12 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !10, baseType: !13)
!13 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "A", file: !3, line: 1, size: 64, align: 64, elements: !14, vtableHolder: !13, identifier: ".?AUA@@")
!14 = !{!15, !21, !25}
!15 = !DIDerivedType(tag: DW_TAG_member, name: "_vptr$A", scope: !3, file: !3, baseType: !16, size: 64, flags: DIFlagArtificial)
!16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !17, size: 64)
!17 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "__vtbl_ptr_type", baseType: !18, size: 64)
!18 = !DISubroutineType(types: !19)
!19 = !{!20}
!20 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!21 = !DISubprogram(name: "f", linkageName: "\01?f@A@@UEAAXXZ", scope: !13, file: !3, line: 2, type: !22, isLocal: false, isDefinition: false, scopeLine: 2, containingType: !13, virtuality: DW_VIRTUALITY_virtual, virtualIndex: 0, flags: DIFlagPrototyped | DIFlagIntroducedVirtual, isOptimized: false)
!22 = !DISubroutineType(types: !23)
!23 = !{null, !24}
!24 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!25 = !DISubprogram(name: "g", linkageName: "\01?g@A@@UEAAXXZ", scope: !13, file: !3, line: 3, type: !22, isLocal: false, isDefinition: false, scopeLine: 3, containingType: !13, virtuality: DW_VIRTUALITY_pure_virtual, virtualIndex: 1, flags: DIFlagPrototyped | DIFlagIntroducedVirtual, isOptimized: false)
!26 = !DISubprogram(name: "f", linkageName: "\01?f@B@@UEAAXXZ", scope: !10, file: !3, line: 6, type: !27, isLocal: false, isDefinition: false, scopeLine: 6, containingType: !10, virtuality: DW_VIRTUALITY_pure_virtual, virtualIndex: 0, flags: DIFlagPrototyped, isOptimized: false)
!27 = !DISubroutineType(types: !28)
!28 = !{null, !29}
!29 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!30 = !DISubprogram(name: "g", linkageName: "\01?g@B@@UEAAXXZ", scope: !10, file: !3, line: 7, type: !27, isLocal: false, isDefinition: false, scopeLine: 7, containingType: !10, virtuality: DW_VIRTUALITY_virtual, virtualIndex: 1, flags: DIFlagPrototyped, isOptimized: false)
!31 = !DISubprogram(name: "f", linkageName: "\01?f@C@@UEAAXXZ", scope: !7, file: !3, line: 10, type: !32, isLocal: false, isDefinition: false, scopeLine: 10, containingType: !7, virtuality: DW_VIRTUALITY_virtual, virtualIndex: 0, flags: DIFlagPrototyped, isOptimized: false)
!32 = !DISubroutineType(types: !33)
!33 = !{null, !34}
!34 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!35 = !DISubprogram(name: "g", linkageName: "\01?g@C@@UEAAXXZ", scope: !7, file: !3, line: 11, type: !32, isLocal: false, isDefinition: false, scopeLine: 11, containingType: !7, virtuality: DW_VIRTUALITY_virtual, virtualIndex: 1, flags: DIFlagPrototyped, isOptimized: false)
!36 = !{i32 2, !"CodeView", i32 1}
!37 = !{i32 2, !"Debug Info Version", i32 3}
!38 = !{i32 1, !"PIC Level", i32 2}
!39 = !{!"clang version 3.9.0 "}
!40 = distinct !DISubprogram(name: "??__Ep@@YAXXZ", scope: !3, file: !3, line: 13, type: !41, isLocal: true, isDefinition: true, scopeLine: 13, flags: DIFlagPrototyped, isOptimized: false, unit: !2, retainedNodes: !4)
!41 = !DISubroutineType(types: !42)
!42 = !{null}
!43 = !DILocation(line: 13, column: 8, scope: !40)
!44 = !DILocation(line: 13, column: 12, scope: !40)
!45 = distinct !DISubprogram(name: "C", linkageName: "\01??0C@@QEAA@XZ", scope: !7, file: !3, line: 9, type: !32, isLocal: false, isDefinition: true, scopeLine: 9, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false, unit: !2, declaration: !46, retainedNodes: !4)
!46 = !DISubprogram(name: "C", scope: !7, type: !32, isLocal: false, isDefinition: false, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false)
!47 = !DILocalVariable(name: "this", arg: 1, scope: !45, type: !6, flags: DIFlagArtificial | DIFlagObjectPointer)
!48 = !DIExpression()
!49 = !DILocation(line: 0, scope: !45)
!50 = !DILocation(line: 9, column: 8, scope: !45)
!51 = distinct !DISubprogram(name: "B", linkageName: "\01??0B@@QEAA@XZ", scope: !10, file: !3, line: 5, type: !27, isLocal: false, isDefinition: true, scopeLine: 5, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false, unit: !2, declaration: !52, retainedNodes: !4)
!52 = !DISubprogram(name: "B", scope: !10, type: !27, isLocal: false, isDefinition: false, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false)
!53 = !DILocalVariable(name: "this", arg: 1, scope: !51, type: !54, flags: DIFlagArtificial | DIFlagObjectPointer)
!54 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64, align: 64)
!55 = !DILocation(line: 0, scope: !51)
!56 = !DILocation(line: 5, column: 8, scope: !51)
!57 = distinct !DISubprogram(name: "A", linkageName: "\01??0A@@QEAA@XZ", scope: !13, file: !3, line: 1, type: !22, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false, unit: !2, declaration: !58, retainedNodes: !4)
!58 = !DISubprogram(name: "A", scope: !13, type: !22, isLocal: false, isDefinition: false, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false)
!59 = !DILocalVariable(name: "this", arg: 1, scope: !57, type: !60, flags: DIFlagArtificial | DIFlagObjectPointer)
!60 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64, align: 64)
!61 = !DILocation(line: 0, scope: !57)
!62 = !DILocation(line: 1, column: 8, scope: !57)
!63 = distinct !DISubprogram(linkageName: "_GLOBAL__sub_I_t.cpp", scope: !3, file: !3, type: !64, isLocal: true, isDefinition: true, flags: DIFlagArtificial, isOptimized: false, unit: !2, retainedNodes: !4)
!64 = !DISubroutineType(types: !4)
!65 = !DILocation(line: 0, scope: !63)

