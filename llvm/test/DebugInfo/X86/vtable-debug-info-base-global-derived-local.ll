; REQUIRES: target={{x86_64.*-linux.*}}
; RUN: llc %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-info %t | FileCheck %s

; Simple inheritance case:
; CBase defined at global scope.
; CDerived defined at function scope.
; For CBase and CDerived we check:
; - Generation of their vtables (including attributes).
; - Generation of their '_vtable$' data members:
;   * Correct scope and attributes

; struct CBase {
;   unsigned B = 1;
;   virtual void zero() {}
;   virtual int one() { return 1; }
; };
;
; int main() {
;   {
;     struct CDerived : CBase {
;       unsigned D = 2;
;       void zero() override {}
;       int one() override { return 11; };
;     };
;
;     {
;       CBase Base;
;       {
;         CDerived Derived;
;       }
;     }
;   }
;
;   return 0;
; }

source_filename = "vtable-debug-info-base-global-derived-local.cpp"
target triple = "x86_64-pc-linux-gnu"

%struct.CBase = type <{ ptr, i32, [4 x i8] }>
%struct.CDerived = type { %struct.CBase.base, i32 }
%struct.CBase.base = type <{ ptr, i32 }>

$_ZN5CBaseC2Ev = comdat any

$_ZN5CBase4zeroEv = comdat any

$_ZN5CBase3oneEv = comdat any

$_ZTV5CBase = comdat any

$_ZTI5CBase = comdat any

$_ZTS5CBase = comdat any

@_ZTV5CBase = linkonce_odr dso_local unnamed_addr constant { [4 x ptr] } { [4 x ptr] [ptr null, ptr @_ZTI5CBase, ptr @_ZN5CBase4zeroEv, ptr @_ZN5CBase3oneEv] }, comdat, align 8, !dbg !0
@_ZTI5CBase = linkonce_odr dso_local constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @_ZTS5CBase }, comdat, align 8
@_ZTVN10__cxxabiv117__class_type_infoE = external global [0 x ptr]
@_ZTS5CBase = linkonce_odr dso_local constant [7 x i8] c"5CBase\00", comdat, align 1
@_ZTVZ4mainE8CDerived = internal unnamed_addr constant { [4 x ptr] } { [4 x ptr] [ptr null, ptr @_ZTIZ4mainE8CDerived, ptr @_ZZ4mainEN8CDerived4zeroEv, ptr @_ZZ4mainEN8CDerived3oneEv] }, align 8, !dbg !5
@_ZTIZ4mainE8CDerived = internal constant { ptr, ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2), ptr @_ZTSZ4mainE8CDerived, ptr @_ZTI5CBase }, align 8
@_ZTVN10__cxxabiv120__si_class_type_infoE = external global [0 x ptr]
@_ZTSZ4mainE8CDerived = internal constant [17 x i8] c"Z4mainE8CDerived\00", align 1

define dso_local noundef i32 @main() #0 !dbg !10 {
entry:
  %retval = alloca i32, align 4
  %Base = alloca %struct.CBase, align 8
  %Derived = alloca %struct.CDerived, align 8
  store i32 0, ptr %retval, align 4
    #dbg_declare(ptr %Base, !48, !DIExpression(), !51)
  call void @_ZN5CBaseC2Ev(ptr noundef nonnull align 8 dereferenceable(12) %Base), !dbg !51
    #dbg_declare(ptr %Derived, !52, !DIExpression(), !54)
  call void @_ZZ4mainEN8CDerivedC2Ev(ptr noundef nonnull align 8 dereferenceable(16) %Derived) , !dbg !54
  ret i32 0
}

define linkonce_odr dso_local void @_ZN5CBaseC2Ev(ptr noundef nonnull align 8 dereferenceable(12) %this) unnamed_addr comdat align 2 {
entry:
  ret void
}

define internal void @_ZZ4mainEN8CDerivedC2Ev(ptr noundef nonnull align 8 dereferenceable(16) %this) unnamed_addr align 2 {
entry:
  ret void
}

define linkonce_odr dso_local void @_ZN5CBase4zeroEv(ptr noundef nonnull align 8 dereferenceable(12) %this) unnamed_addr comdat align 2 {
entry:
  ret void
}

define linkonce_odr dso_local noundef i32 @_ZN5CBase3oneEv(ptr noundef nonnull align 8 dereferenceable(12) %this) unnamed_addr comdat align 2 {
entry:
  ret i32 1
}

define internal void @_ZZ4mainEN8CDerived4zeroEv(ptr noundef nonnull align 8 dereferenceable(16) %this) unnamed_addr align 2 {
entry:
  ret void
}

define internal noundef i32 @_ZZ4mainEN8CDerived3oneEv(ptr noundef nonnull align 8 dereferenceable(16) %this) unnamed_addr align 2 {
entry:
  ret i32 11
}

attributes #0 = { mustprogress noinline norecurse nounwind uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!40, !41}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "_vtable$", linkageName: "_ZTV5CBase", scope: !2, file: !3, type: !7, isLocal: false, isDefinition: true, declaration: !39, align: 64)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, emissionKind: FullDebug, globals: !4)
!3 = !DIFile(filename: "vtable-debug-info-base-global-derived-local.cpp", directory: "")
!4 = !{!0, !5}
!5 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression())
!6 = distinct !DIGlobalVariable(name: "_vtable$", linkageName: "_ZTVZ4mainE8CDerived", scope: !2, file: !3, type: !7, isLocal: true, isDefinition: true, declaration: !8, align: 64)
!7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!8 = !DIDerivedType(tag: DW_TAG_variable, name: "_vtable$", scope: !9, file: !3, baseType: !7, flags: DIFlagPrivate | DIFlagArtificial | DIFlagStaticMember)
!9 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "CDerived", scope: !10, file: !3, line: 9, size: 128, flags: DIFlagTypePassByReference | DIFlagNonTrivial, elements: !15, vtableHolder: !17)
!10 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 7, type: !11, scopeLine: 7, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !14)
!11 = !DISubroutineType(types: !12)
!12 = !{!13}
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !{}
!15 = !{!16}
!16 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !9, baseType: !17, extraData: i32 0)
!17 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "CBase", file: !3, line: 1, size: 128, flags: DIFlagTypePassByReference | DIFlagNonTrivial, elements: !18, vtableHolder: !17, identifier: "_ZTS5CBase")
!18 = !{}
!39 = !DIDerivedType(tag: DW_TAG_variable, name: "_vtable$", scope: !17, file: !3, baseType: !7, flags: DIFlagPrivate | DIFlagArtificial | DIFlagStaticMember)
!40 = !{i32 7, !"Dwarf Version", i32 5}
!41 = !{i32 2, !"Debug Info Version", i32 3}
!48 = !DILocalVariable(name: "Base", scope: !49, file: !3, line: 16, type: !17)
!49 = distinct !DILexicalBlock(scope: !50, file: !3, line: 15, column: 5)
!50 = distinct !DILexicalBlock(scope: !10, file: !3, line: 8, column: 3)
!51 = !DILocation(line: 16, column: 13, scope: !49)
!52 = !DILocalVariable(name: "Derived", scope: !53, file: !3, line: 18, type: !9)
!53 = distinct !DILexicalBlock(scope: !49, file: !3, line: 17, column: 7)
!54 = !DILocation(line: 18, column: 18, scope: !53)

; CHECK:     .debug_info contents:
; CHECK-NEXT: 0x00000000:     Compile Unit:
; CHECK: {{.*}}DW_TAG_variable
; CHECK-NEXT: DW_AT_specification ([[VARDIE_1:.+]] "_vtable$")
; CHECK-NEXT: DW_AT_alignment	(8)
; CHECK-NEXT: DW_AT_location (DW_OP_addrx 0x0)
; CHECK-NEXT: DW_AT_linkage_name ("_ZTV5CBase")

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
; CHECK-NEXT: DW_AT_linkage_name ("_ZTVZ4mainE8CDerived")

; CHECK: {{.*}}DW_TAG_subprogram
; CHECK-NEXT: DW_AT_low_pc
; CHECK-NEXT: DW_AT_high_pc
; CHECK-NEXT: DW_AT_frame_base
; CHECK-NEXT: DW_AT_name ("main")

; CHECK: {{.*}}DW_TAG_structure_type
; CHECK: DW_AT_name	("CDerived")

; CHECK: [[VARDIE_2]]: DW_TAG_variable
; CHECK-NEXT: DW_AT_name ("_vtable$")
; CHECK-NEXT: DW_AT_type ({{.*}} "void *")
; CHECK: DW_AT_artificial	(true)
; CHECK-NEXT: DW_AT_accessibility	(DW_ACCESS_private)
