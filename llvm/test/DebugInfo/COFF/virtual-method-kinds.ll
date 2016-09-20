; RUN: llc < %s -filetype=obj | llvm-readobj - -codeview | FileCheck %s

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
; CHECK-NEXT:   AccessSpecifier: Public (0x3)
; CHECK-NEXT:   MethodKind: Virtual (0x1)
; CHECK-NEXT:   Type: void C::() ({{.*}})
; CHECK-NEXT:   Name: f
; CHECK-NEXT: }
; CHECK-NEXT: OneMethod {
; CHECK-NEXT:   AccessSpecifier: Public (0x3)
; CHECK-NEXT:   MethodKind: Virtual (0x1)
; CHECK-NEXT:   Type: void C::() ({{.*}})
; CHECK-NEXT:   Name: g
; CHECK-NEXT: }

; CHECK:      OneMethod {
; CHECK-NEXT:   AccessSpecifier: Public (0x3)
; CHECK-NEXT:   MethodKind: PureVirtual (0x5)
; CHECK-NEXT:   Type: void B::() ({{.*}})
; CHECK-NEXT:   Name: f
; CHECK-NEXT: }
; CHECK-NEXT: OneMethod {
; CHECK-NEXT:   AccessSpecifier: Public (0x3)
; CHECK-NEXT:   MethodKind: Virtual (0x1)
; CHECK-NEXT:   Type: void B::() ({{.*}})
; CHECK-NEXT:   Name: g
; CHECK-NEXT: }

; CHECK:      OneMethod {
; CHECK-NEXT:   AccessSpecifier: Public (0x3)
; CHECK-NEXT:   MethodKind: IntroducingVirtual (0x4)
; CHECK-NEXT:   Type: void A::() ({{.*}})
; CHECK-NEXT:   VFTableOffset: 0x0
; CHECK-NEXT:   Name: f
; CHECK-NEXT: }
; CHECK-NEXT: OneMethod {
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
%struct.A = type { i32 (...)** }
%rtti.CompleteObjectLocator = type { i32, i32, i32, i32, i32, i32 }
%rtti.TypeDescriptor7 = type { i8**, i8*, [8 x i8] }
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

@"\01?p@@3PEAUC@@EA" = global %struct.C* null, align 8, !dbg !4
@0 = private unnamed_addr constant [3 x i8*] [i8* bitcast (%rtti.CompleteObjectLocator* @"\01??_R4C@@6B@" to i8*), i8* bitcast (void (%struct.C*)* @"\01?f@C@@UEAAXXZ" to i8*), i8* bitcast (void (%struct.C*)* @"\01?g@C@@UEAAXXZ" to i8*)], comdat($"\01??_7C@@6B@")
@"\01??_R4C@@6B@" = linkonce_odr constant %rtti.CompleteObjectLocator { i32 1, i32 0, i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.TypeDescriptor7* @"\01??_R0?AUC@@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.ClassHierarchyDescriptor* @"\01??_R3C@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.CompleteObjectLocator* @"\01??_R4C@@6B@" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32) }, comdat
@"\01??_7type_info@@6B@" = external constant i8*
@"\01??_R0?AUC@@@8" = linkonce_odr global %rtti.TypeDescriptor7 { i8** @"\01??_7type_info@@6B@", i8* null, [8 x i8] c".?AUC@@\00" }, comdat
@__ImageBase = external constant i8
@"\01??_R3C@@8" = linkonce_odr constant %rtti.ClassHierarchyDescriptor { i32 0, i32 0, i32 3, i32 trunc (i64 sub nuw nsw (i64 ptrtoint ([4 x i32]* @"\01??_R2C@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32) }, comdat
@"\01??_R2C@@8" = linkonce_odr constant [4 x i32] [i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.BaseClassDescriptor* @"\01??_R1A@?0A@EA@C@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.BaseClassDescriptor* @"\01??_R1A@?0A@EA@B@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.BaseClassDescriptor* @"\01??_R1A@?0A@EA@A@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 0], comdat
@"\01??_R1A@?0A@EA@C@@8" = linkonce_odr constant %rtti.BaseClassDescriptor { i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.TypeDescriptor7* @"\01??_R0?AUC@@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 2, i32 0, i32 -1, i32 0, i32 64, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.ClassHierarchyDescriptor* @"\01??_R3C@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32) }, comdat
@"\01??_R1A@?0A@EA@B@@8" = linkonce_odr constant %rtti.BaseClassDescriptor { i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.TypeDescriptor7* @"\01??_R0?AUB@@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 1, i32 0, i32 -1, i32 0, i32 64, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.ClassHierarchyDescriptor* @"\01??_R3B@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32) }, comdat
@"\01??_R0?AUB@@@8" = linkonce_odr global %rtti.TypeDescriptor7 { i8** @"\01??_7type_info@@6B@", i8* null, [8 x i8] c".?AUB@@\00" }, comdat
@"\01??_R3B@@8" = linkonce_odr constant %rtti.ClassHierarchyDescriptor { i32 0, i32 0, i32 2, i32 trunc (i64 sub nuw nsw (i64 ptrtoint ([3 x i32]* @"\01??_R2B@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32) }, comdat
@"\01??_R2B@@8" = linkonce_odr constant [3 x i32] [i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.BaseClassDescriptor* @"\01??_R1A@?0A@EA@B@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.BaseClassDescriptor* @"\01??_R1A@?0A@EA@A@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 0], comdat
@"\01??_R1A@?0A@EA@A@@8" = linkonce_odr constant %rtti.BaseClassDescriptor { i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.TypeDescriptor7* @"\01??_R0?AUA@@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 0, i32 0, i32 -1, i32 0, i32 64, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.ClassHierarchyDescriptor* @"\01??_R3A@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32) }, comdat
@"\01??_R0?AUA@@@8" = linkonce_odr global %rtti.TypeDescriptor7 { i8** @"\01??_7type_info@@6B@", i8* null, [8 x i8] c".?AUA@@\00" }, comdat
@"\01??_R3A@@8" = linkonce_odr constant %rtti.ClassHierarchyDescriptor { i32 0, i32 0, i32 1, i32 trunc (i64 sub nuw nsw (i64 ptrtoint ([2 x i32]* @"\01??_R2A@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32) }, comdat
@"\01??_R2A@@8" = linkonce_odr constant [2 x i32] [i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.BaseClassDescriptor* @"\01??_R1A@?0A@EA@A@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 0], comdat
@1 = private unnamed_addr constant [3 x i8*] [i8* bitcast (%rtti.CompleteObjectLocator* @"\01??_R4B@@6B@" to i8*), i8* bitcast (void ()* @_purecall to i8*), i8* bitcast (void (%struct.B*)* @"\01?g@B@@UEAAXXZ" to i8*)], comdat($"\01??_7B@@6B@")
@"\01??_R4B@@6B@" = linkonce_odr constant %rtti.CompleteObjectLocator { i32 1, i32 0, i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.TypeDescriptor7* @"\01??_R0?AUB@@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.ClassHierarchyDescriptor* @"\01??_R3B@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.CompleteObjectLocator* @"\01??_R4B@@6B@" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32) }, comdat
@2 = private unnamed_addr constant [3 x i8*] [i8* bitcast (%rtti.CompleteObjectLocator* @"\01??_R4A@@6B@" to i8*), i8* bitcast (void (%struct.A*)* @"\01?f@A@@UEAAXXZ" to i8*), i8* bitcast (void ()* @_purecall to i8*)], comdat($"\01??_7A@@6B@")
@"\01??_R4A@@6B@" = linkonce_odr constant %rtti.CompleteObjectLocator { i32 1, i32 0, i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.TypeDescriptor7* @"\01??_R0?AUA@@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.ClassHierarchyDescriptor* @"\01??_R3A@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.CompleteObjectLocator* @"\01??_R4A@@6B@" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32) }, comdat
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_t.cpp, i8* null }]

@"\01??_7C@@6B@" = unnamed_addr alias i8*, getelementptr inbounds ([3 x i8*], [3 x i8*]* @0, i32 0, i32 1)
@"\01??_7B@@6B@" = unnamed_addr alias i8*, getelementptr inbounds ([3 x i8*], [3 x i8*]* @1, i32 0, i32 1)
@"\01??_7A@@6B@" = unnamed_addr alias i8*, getelementptr inbounds ([3 x i8*], [3 x i8*]* @2, i32 0, i32 1)

; Function Attrs: uwtable
define internal void @"\01??__Ep@@YAXXZ"() #0 !dbg !39 {
entry:
  %call = call i8* @"\01??2@YAPEAX_K@Z"(i64 8) #5, !dbg !42
  %0 = bitcast i8* %call to %struct.C*, !dbg !42
  %call1 = call %struct.C* @"\01??0C@@QEAA@XZ"(%struct.C* %0) #6, !dbg !43
  store %struct.C* %0, %struct.C** @"\01?p@@3PEAUC@@EA", align 8, !dbg !42
  ret void, !dbg !43
}

; Function Attrs: nobuiltin
declare noalias i8* @"\01??2@YAPEAX_K@Z"(i64) #1

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr %struct.C* @"\01??0C@@QEAA@XZ"(%struct.C* returned %this) unnamed_addr #2 comdat align 2 !dbg !44 {
entry:
  %this.addr = alloca %struct.C*, align 8
  store %struct.C* %this, %struct.C** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.C** %this.addr, metadata !46, metadata !47), !dbg !48
  %this1 = load %struct.C*, %struct.C** %this.addr, align 8
  %0 = bitcast %struct.C* %this1 to %struct.B*, !dbg !49
  %call = call %struct.B* @"\01??0B@@QEAA@XZ"(%struct.B* %0) #6, !dbg !49
  %1 = bitcast %struct.C* %this1 to i32 (...)***, !dbg !49
  store i32 (...)** bitcast (i8** @"\01??_7C@@6B@" to i32 (...)**), i32 (...)*** %1, align 8, !dbg !49
  ret %struct.C* %this1, !dbg !49
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #3

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr %struct.B* @"\01??0B@@QEAA@XZ"(%struct.B* returned %this) unnamed_addr #2 comdat align 2 !dbg !50 {
entry:
  %this.addr = alloca %struct.B*, align 8
  store %struct.B* %this, %struct.B** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.B** %this.addr, metadata !52, metadata !47), !dbg !54
  %this1 = load %struct.B*, %struct.B** %this.addr, align 8
  %0 = bitcast %struct.B* %this1 to %struct.A*, !dbg !55
  %call = call %struct.A* @"\01??0A@@QEAA@XZ"(%struct.A* %0) #6, !dbg !55
  %1 = bitcast %struct.B* %this1 to i32 (...)***, !dbg !55
  store i32 (...)** bitcast (i8** @"\01??_7B@@6B@" to i32 (...)**), i32 (...)*** %1, align 8, !dbg !55
  ret %struct.B* %this1, !dbg !55
}

declare void @"\01?f@C@@UEAAXXZ"(%struct.C*) unnamed_addr #4

declare void @"\01?g@C@@UEAAXXZ"(%struct.C*) unnamed_addr #4

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr %struct.A* @"\01??0A@@QEAA@XZ"(%struct.A* returned %this) unnamed_addr #2 comdat align 2 !dbg !56 {
entry:
  %this.addr = alloca %struct.A*, align 8
  store %struct.A* %this, %struct.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.A** %this.addr, metadata !58, metadata !47), !dbg !60
  %this1 = load %struct.A*, %struct.A** %this.addr, align 8
  %0 = bitcast %struct.A* %this1 to i32 (...)***, !dbg !61
  store i32 (...)** bitcast (i8** @"\01??_7A@@6B@" to i32 (...)**), i32 (...)*** %0, align 8, !dbg !61
  ret %struct.A* %this1, !dbg !61
}

declare void @_purecall() unnamed_addr

declare void @"\01?g@B@@UEAAXXZ"(%struct.B*) unnamed_addr #4

declare void @"\01?f@A@@UEAAXXZ"(%struct.A*) unnamed_addr #4

; Function Attrs: uwtable
define internal void @_GLOBAL__sub_I_t.cpp() #0 !dbg !62 {
entry:
  call void @"\01??__Ep@@YAXXZ"(), !dbg !64
  ret void
}

attributes #0 = { uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nobuiltin "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { inlinehint nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind readnone }
attributes #4 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { builtin }
attributes #6 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!35, !36, !37}
!llvm.ident = !{!38}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.9.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !3)
!1 = !DIFile(filename: "t.cpp", directory: "D:\5Csrc\5Cllvm\5Cbuild")
!2 = !{}
!3 = !{!4}
!4 = distinct !DIGlobalVariable(name: "p", linkageName: "\01?p@@3PEAUC@@EA", scope: !0, file: !1, line: 13, type: !5, isLocal: false, isDefinition: true)
!5 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 64, align: 64)
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "C", file: !1, line: 9, size: 64, align: 64, elements: !7, vtableHolder: !12, identifier: ".?AUC@@")
!7 = !{!8, !30, !34}
!8 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !6, baseType: !9)
!9 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "B", file: !1, line: 5, size: 64, align: 64, elements: !10, vtableHolder: !12, identifier: ".?AUB@@")
!10 = !{!11, !25, !29}
!11 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !9, baseType: !12)
!12 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "A", file: !1, line: 1, size: 64, align: 64, elements: !13, vtableHolder: !12, identifier: ".?AUA@@")
!13 = !{!14, !20, !24}
!14 = !DIDerivedType(tag: DW_TAG_member, name: "_vptr$A", scope: !1, file: !1, baseType: !15, size: 64, flags: DIFlagArtificial)
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !16, size: 64)
!16 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "__vtbl_ptr_type", baseType: !17, size: 64)
!17 = !DISubroutineType(types: !18)
!18 = !{!19}
!19 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!20 = !DISubprogram(name: "f", linkageName: "\01?f@A@@UEAAXXZ", scope: !12, file: !1, line: 2, type: !21, isLocal: false, isDefinition: false, scopeLine: 2, containingType: !12, virtuality: DW_VIRTUALITY_virtual, virtualIndex: 0, flags: DIFlagPrototyped | DIFlagIntroducedVirtual, isOptimized: false)
!21 = !DISubroutineType(types: !22)
!22 = !{null, !23}
!23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!24 = !DISubprogram(name: "g", linkageName: "\01?g@A@@UEAAXXZ", scope: !12, file: !1, line: 3, type: !21, isLocal: false, isDefinition: false, scopeLine: 3, containingType: !12, virtuality: DW_VIRTUALITY_pure_virtual, virtualIndex: 1, flags: DIFlagPrototyped | DIFlagIntroducedVirtual, isOptimized: false)
!25 = !DISubprogram(name: "f", linkageName: "\01?f@B@@UEAAXXZ", scope: !9, file: !1, line: 6, type: !26, isLocal: false, isDefinition: false, scopeLine: 6, containingType: !9, virtuality: DW_VIRTUALITY_pure_virtual, virtualIndex: 0, flags: DIFlagPrototyped, isOptimized: false)
!26 = !DISubroutineType(types: !27)
!27 = !{null, !28}
!28 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!29 = !DISubprogram(name: "g", linkageName: "\01?g@B@@UEAAXXZ", scope: !9, file: !1, line: 7, type: !26, isLocal: false, isDefinition: false, scopeLine: 7, containingType: !9, virtuality: DW_VIRTUALITY_virtual, virtualIndex: 1, flags: DIFlagPrototyped, isOptimized: false)
!30 = !DISubprogram(name: "f", linkageName: "\01?f@C@@UEAAXXZ", scope: !6, file: !1, line: 10, type: !31, isLocal: false, isDefinition: false, scopeLine: 10, containingType: !6, virtuality: DW_VIRTUALITY_virtual, virtualIndex: 0, flags: DIFlagPrototyped, isOptimized: false)
!31 = !DISubroutineType(types: !32)
!32 = !{null, !33}
!33 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!34 = !DISubprogram(name: "g", linkageName: "\01?g@C@@UEAAXXZ", scope: !6, file: !1, line: 11, type: !31, isLocal: false, isDefinition: false, scopeLine: 11, containingType: !6, virtuality: DW_VIRTUALITY_virtual, virtualIndex: 1, flags: DIFlagPrototyped, isOptimized: false)
!35 = !{i32 2, !"CodeView", i32 1}
!36 = !{i32 2, !"Debug Info Version", i32 3}
!37 = !{i32 1, !"PIC Level", i32 2}
!38 = !{!"clang version 3.9.0 "}
!39 = distinct !DISubprogram(name: "??__Ep@@YAXXZ", scope: !1, file: !1, line: 13, type: !40, isLocal: true, isDefinition: true, scopeLine: 13, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!40 = !DISubroutineType(types: !41)
!41 = !{null}
!42 = !DILocation(line: 13, column: 8, scope: !39)
!43 = !DILocation(line: 13, column: 12, scope: !39)
!44 = distinct !DISubprogram(name: "C", linkageName: "\01??0C@@QEAA@XZ", scope: !6, file: !1, line: 9, type: !31, isLocal: false, isDefinition: true, scopeLine: 9, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false, unit: !0, declaration: !45, variables: !2)
!45 = !DISubprogram(name: "C", scope: !6, type: !31, isLocal: false, isDefinition: false, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false)
!46 = !DILocalVariable(name: "this", arg: 1, scope: !44, type: !5, flags: DIFlagArtificial | DIFlagObjectPointer)
!47 = !DIExpression()
!48 = !DILocation(line: 0, scope: !44)
!49 = !DILocation(line: 9, column: 8, scope: !44)
!50 = distinct !DISubprogram(name: "B", linkageName: "\01??0B@@QEAA@XZ", scope: !9, file: !1, line: 5, type: !26, isLocal: false, isDefinition: true, scopeLine: 5, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false, unit: !0, declaration: !51, variables: !2)
!51 = !DISubprogram(name: "B", scope: !9, type: !26, isLocal: false, isDefinition: false, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false)
!52 = !DILocalVariable(name: "this", arg: 1, scope: !50, type: !53, flags: DIFlagArtificial | DIFlagObjectPointer)
!53 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 64, align: 64)
!54 = !DILocation(line: 0, scope: !50)
!55 = !DILocation(line: 5, column: 8, scope: !50)
!56 = distinct !DISubprogram(name: "A", linkageName: "\01??0A@@QEAA@XZ", scope: !12, file: !1, line: 1, type: !21, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false, unit: !0, declaration: !57, variables: !2)
!57 = !DISubprogram(name: "A", scope: !12, type: !21, isLocal: false, isDefinition: false, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false)
!58 = !DILocalVariable(name: "this", arg: 1, scope: !56, type: !59, flags: DIFlagArtificial | DIFlagObjectPointer)
!59 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64, align: 64)
!60 = !DILocation(line: 0, scope: !56)
!61 = !DILocation(line: 1, column: 8, scope: !56)
!62 = distinct !DISubprogram(linkageName: "_GLOBAL__sub_I_t.cpp", scope: !1, file: !1, type: !63, isLocal: true, isDefinition: true, flags: DIFlagArtificial, isOptimized: false, unit: !0, variables: !2)
!63 = !DISubroutineType(types: !2)
!64 = !DILocation(line: 0, scope: !62)
