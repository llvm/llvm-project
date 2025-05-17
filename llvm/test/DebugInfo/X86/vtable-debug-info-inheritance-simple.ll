; REQUIRES: target={{x86_64.*-linux.*}}
; RUN: llc %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-info %t | FileCheck %s

; Simple inheritance case:
; For CBase and CDerived we check:
; - Generation of their vtables (including attributes).
; - Generation of their '_vtable$' data members:
;   * Correct scope and attributes

; namespace NSP {
;   struct CBase {
;     unsigned B = 1;
;     virtual void zero() {}
;     virtual int one() { return 1; }
;     virtual int two() { return 2; }
;     virtual int three() { return 3; }
;   };
; }
;
; struct CDerived : NSP::CBase {
;   unsigned D = 2;
;   void zero() override {}
;   int two() override { return 22; };
;   int three() override { return 33; }
; };
;
; int main() {
;   NSP::CBase Base;
;   CDerived Derived;
;
;   return 0;
; }

source_filename = "vtable-debug-info-inheritance-simple.cpp"
target triple = "x86_64-linux"

%"struct.NSP::CBase" = type <{ ptr, i32, [4 x i8] }>
%struct.CDerived = type { %"struct.NSP::CBase.base", i32 }
%"struct.NSP::CBase.base" = type <{ ptr, i32 }>

$_ZN3NSP5CBaseC2Ev = comdat any

$_ZN8CDerivedC2Ev = comdat any

$_ZN3NSP5CBase4zeroEv = comdat any

$_ZN3NSP5CBase3oneEv = comdat any

$_ZN3NSP5CBase3twoEv = comdat any

$_ZN3NSP5CBase5threeEv = comdat any

$_ZN8CDerived4zeroEv = comdat any

$_ZN8CDerived3twoEv = comdat any

$_ZN8CDerived5threeEv = comdat any

$_ZTVN3NSP5CBaseE = comdat any

$_ZTSN3NSP5CBaseE = comdat any

$_ZTIN3NSP5CBaseE = comdat any

$_ZTV8CDerived = comdat any

$_ZTS8CDerived = comdat any

$_ZTI8CDerived = comdat any

@_ZTVN3NSP5CBaseE = linkonce_odr dso_local unnamed_addr constant { [6 x ptr] } { [6 x ptr] [ptr null, ptr @_ZTIN3NSP5CBaseE, ptr @_ZN3NSP5CBase4zeroEv, ptr @_ZN3NSP5CBase3oneEv, ptr @_ZN3NSP5CBase3twoEv, ptr @_ZN3NSP5CBase5threeEv] }, comdat, align 8, !dbg !0
@_ZTVN10__cxxabiv117__class_type_infoE = external global [0 x ptr]
@_ZTSN3NSP5CBaseE = linkonce_odr dso_local constant [13 x i8] c"N3NSP5CBaseE\00", comdat, align 1
@_ZTIN3NSP5CBaseE = linkonce_odr dso_local constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @_ZTSN3NSP5CBaseE }, comdat, align 8
@_ZTV8CDerived = linkonce_odr dso_local unnamed_addr constant { [6 x ptr] } { [6 x ptr] [ptr null, ptr @_ZTI8CDerived, ptr @_ZN8CDerived4zeroEv, ptr @_ZN3NSP5CBase3oneEv, ptr @_ZN8CDerived3twoEv, ptr @_ZN8CDerived5threeEv] }, comdat, align 8, !dbg !5
@_ZTVN10__cxxabiv120__si_class_type_infoE = external global [0 x ptr]
@_ZTS8CDerived = linkonce_odr dso_local constant [10 x i8] c"8CDerived\00", comdat, align 1
@_ZTI8CDerived = linkonce_odr dso_local constant { ptr, ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2), ptr @_ZTS8CDerived, ptr @_ZTIN3NSP5CBaseE }, comdat, align 8

define dso_local noundef i32 @main() #0 !dbg !51 {
entry:
  %retval = alloca i32, align 4
  %Base = alloca %"struct.NSP::CBase", align 8
  %Derived = alloca %struct.CDerived, align 8
  store i32 0, ptr %retval, align 4
    #dbg_declare(ptr %Base, !53, !DIExpression(), !54)
  call void @_ZN3NSP5CBaseC2Ev(ptr noundef nonnull align 8 dereferenceable(12) %Base), !dbg !54
    #dbg_declare(ptr %Derived, !55, !DIExpression(), !56)
  call void @_ZN8CDerivedC2Ev(ptr noundef nonnull align 8 dereferenceable(16) %Derived), !dbg !56
  ret i32 0
}

define linkonce_odr dso_local void @_ZN3NSP5CBaseC2Ev(ptr noundef nonnull align 8 dereferenceable(12) %this) unnamed_addr #1 comdat align 2 {
entry:
  ret void
}

define linkonce_odr dso_local void @_ZN8CDerivedC2Ev(ptr noundef nonnull align 8 dereferenceable(16) %this) unnamed_addr comdat align 2 {
entry:
  ret void
}

define linkonce_odr dso_local void @_ZN3NSP5CBase4zeroEv(ptr noundef nonnull align 8 dereferenceable(12) %this) unnamed_addr comdat align 2 {
entry:
  ret void
}

define linkonce_odr dso_local noundef i32 @_ZN3NSP5CBase3oneEv(ptr noundef nonnull align 8 dereferenceable(12) %this) unnamed_addr comdat align 2 {
entry:
  ret i32 1
}

define linkonce_odr dso_local noundef i32 @_ZN3NSP5CBase3twoEv(ptr noundef nonnull align 8 dereferenceable(12) %this) unnamed_addr comdat align 2 {
entry:
  ret i32 2
}

define linkonce_odr dso_local noundef i32 @_ZN3NSP5CBase5threeEv(ptr noundef nonnull align 8 dereferenceable(12) %this) unnamed_addr comdat align 2 {
entry:
  ret i32 3
}

define linkonce_odr dso_local void @_ZN8CDerived4zeroEv(ptr noundef nonnull align 8 dereferenceable(16) %this) unnamed_addr comdat align 2 {
entry:
  ret void
}

define linkonce_odr dso_local noundef i32 @_ZN8CDerived3twoEv(ptr noundef nonnull align 8 dereferenceable(16) %this) unnamed_addr comdat align 2 {
entry:
  ret i32 22
}

define linkonce_odr dso_local noundef i32 @_ZN8CDerived5threeEv(ptr noundef nonnull align 8 dereferenceable(16) %this) unnamed_addr comdat align 2 {
entry:
  ret i32 33
}

attributes #0 = { mustprogress noinline norecurse nounwind uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!43, !44}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "_vtable$", linkageName: "_ZTVN3NSP5CBaseE", scope: !2, file: !7, type: !8, isLocal: false, isDefinition: true, declaration: !42, align: 64)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, emissionKind: FullDebug, globals: !4)
!3 = !DIFile(filename: "vtable-debug-info-inheritance-simple.cpp", directory: "")
!4 = !{!0, !5}
!5 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression())
!6 = distinct !DIGlobalVariable(name: "_vtable$", linkageName: "_ZTV8CDerived", scope: !2, file: !7, type: !8, isLocal: false, isDefinition: true, declaration: !9, align: 64)
!7 = !DIFile(filename: "vtable-debug-info-inheritance-simple.cpp", directory: "")
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!9 = !DIDerivedType(tag: DW_TAG_variable, name: "_vtable$", scope: !10, file: !7, baseType: !8, flags: DIFlagPrivate | DIFlagArtificial | DIFlagStaticMember)
!10 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "CDerived", file: !7, line: 19, size: 128, flags: DIFlagTypePassByReference | DIFlagNonTrivial, elements: !11, vtableHolder: !13, identifier: "_ZTS8CDerived")
!11 = !{!12}
!12 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !10, baseType: !13, extraData: i32 0)
!13 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "CBase", scope: !14, file: !7, line: 10, size: 128, flags: DIFlagTypePassByReference | DIFlagNonTrivial, elements: !15, vtableHolder: !13, identifier: "_ZTSN3NSP5CBaseE")
!14 = !DINamespace(name: "NSP", scope: null)
!15 = !{}
!19 = !DISubroutineType(types: !20)
!20 = !{!21}
!21 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!42 = !DIDerivedType(tag: DW_TAG_variable, name: "_vtable$", scope: !13, file: !7, baseType: !8, flags: DIFlagPrivate | DIFlagArtificial | DIFlagStaticMember)
!43 = !{i32 7, !"Dwarf Version", i32 5}
!44 = !{i32 2, !"Debug Info Version", i32 3}
!51 = distinct !DISubprogram(name: "main", scope: !7, file: !7, line: 26, type: !19, scopeLine: 26, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !52)
!52 = !{}
!53 = !DILocalVariable(name: "Base", scope: !51, file: !7, line: 27, type: !13)
!54 = !DILocation(line: 27, column: 14, scope: !51)
!55 = !DILocalVariable(name: "Derived", scope: !51, file: !7, line: 28, type: !10)
!56 = !DILocation(line: 28, column: 12, scope: !51)

; CHECK:     .debug_info contents:
; CHECK-NEXT: 0x00000000:     Compile Unit:
; CHECK: {{.*}}DW_TAG_variable
; CHECK-NEXT: DW_AT_specification ([[VARDIE_1:.+]] "_vtable$")
; CHECK-NEXT: DW_AT_alignment	(8)
; CHECK-NEXT: DW_AT_location (DW_OP_addrx 0x0)
; CHECK-NEXT: DW_AT_linkage_name ("_ZTVN3NSP5CBaseE")

; CHECK: {{.*}}DW_TAG_namespace
; CHECK-NEXT: DW_AT_name ("NSP")

; CHECK: {{.*}}DW_TAG_structure_type
; CHECK: DW_AT_name	("CBase")

; CHECK: [[VARDIE_1]]: DW_TAG_variable
; CHECK-NEXT: DW_AT_name ("_vtable$")
; CHECK-NEXT: DW_AT_type ({{.*}} "void *")
; CHECK: DW_AT_artificial	(true)
; CHECK-NEXT: DW_AT_accessibility	(DW_ACCESS_private)

; CHECK: {{.*}}DW_TAG_variable
; CHECK-NEXT: DW_AT_specification ([[VARDIE_2:.+]] "_vtable$")
; CHECK-NEXT: DW_AT_alignment	(8)
; CHECK-NEXT: DW_AT_location (DW_OP_addrx 0x1)
; CHECK-NEXT: DW_AT_linkage_name ("_ZTV8CDerived")

; CHECK: {{.*}}DW_TAG_structure_type
; CHECK: DW_AT_name	("CDerived")

; CHECK: [[VARDIE_2]]: DW_TAG_variable
; CHECK-NEXT: DW_AT_name ("_vtable$")
; CHECK-NEXT: DW_AT_type ({{.*}} "void *")
; CHECK: DW_AT_artificial	(true)
; CHECK-NEXT: DW_AT_accessibility	(DW_ACCESS_private)
