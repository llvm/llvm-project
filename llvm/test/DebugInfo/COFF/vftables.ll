; RUN: llc < %s -filetype=obj | llvm-readobj - -codeview | FileCheck %s

; C++ source to regenerate:
; struct A {
;   virtual void f();
;   virtual void g();
;   int a = 0;
; };
; struct B {
;   virtual void g();
;   virtual void f();
;   int b = 0;
; };
; struct C : A, B {
;   virtual void g();
;   virtual void f();
;   int c = 0;
; };
; struct D : C {
;   virtual void g();
;   virtual void f();
;   int d = 0;
; };
; void h() { D d; }

; CHECK:       VFTableShape ([[vshape_1:0x[A-Z0-9]+]]) {
; CHECK-NEXT:    TypeLeafKind: LF_VTSHAPE (0xA)
; CHECK-NEXT:    VFEntryCount: 1
; CHECK-NEXT:  }

; CHECK:       Pointer ([[vptr_1:0x[A-Z0-9]+]]) {
; CHECK-NEXT:    TypeLeafKind: LF_POINTER (0x1002)
; CHECK-NEXT:    PointeeType: <vftable 1 methods> ([[vshape_1]])

; CHECK:       FieldList ([[a_members:0x[A-Z0-9]+]]) {
; CHECK-NEXT:    TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK-NEXT:    VFPtr {
; CHECK-NEXT:      TypeLeafKind: LF_VFUNCTAB (0x1409)
; CHECK-NEXT:      Type: <vftable 1 methods>* ([[vptr_1]])
; CHECK-NEXT:    }
; CHECK-NEXT:    DataMember {
; CHECK-NEXT:      TypeLeafKind: LF_MEMBER (0x150D)
; CHECK-NEXT:      AccessSpecifier: Public (0x3)
; CHECK-NEXT:      Type: int (0x74)
; CHECK-NEXT:      FieldOffset: 0x8
; CHECK-NEXT:      Name: a
; CHECK-NEXT:    }
; CHECK-NEXT:    OneMethod {
; CHECK-NEXT:      TypeLeafKind: LF_ONEMETHOD (0x1511)
; CHECK-NEXT:      AccessSpecifier: Public (0x3)
; CHECK-NEXT:      MethodKind: IntroducingVirtual (0x4)
; CHECK-NEXT:      Type: void A::()
; CHECK-NEXT:      VFTableOffset: 0x0
; CHECK-NEXT:      Name: g
; CHECK-NEXT:    }
; CHECK-NEXT:  }

; CHECK:        Struct ({{.*}}) {
; CHECK-NEXT:     TypeLeafKind: LF_STRUCTURE (0x1505)
; CHECK-NEXT:     MemberCount: 3
; CHECK-NEXT:     Properties [ (0x200)
; CHECK-NEXT:       HasUniqueName (0x200)
; CHECK-NEXT:     ]
; CHECK-NEXT:     FieldList: <field list> ([[a_members]])
; CHECK-NEXT:     DerivedFrom: 0x0
; CHECK-NEXT:     VShape: <vftable 1 methods> ([[vshape_1]])
; CHECK-NEXT:     SizeOf: 16
; CHECK-NEXT:     Name: A
; CHECK-NEXT:     LinkageName: .?AUA@@
; CHECK-NEXT:   }

; CHECK:       VFTableShape ([[vshape_2:0x[A-Z0-9]+]]) {
; CHECK-NEXT:    TypeLeafKind: LF_VTSHAPE (0xA)
; CHECK-NEXT:    VFEntryCount: 2
; CHECK-NEXT:  }

; CHECK:       Pointer ([[vptr_2:0x[A-Z0-9]+]]) {
; CHECK-NEXT:    TypeLeafKind: LF_POINTER (0x1002)
; CHECK-NEXT:    PointeeType: <vftable 2 methods> ([[vshape_2]])

; CHECK:       FieldList ([[b_members:0x[A-Z0-9]+]]) {
; CHECK-NEXT:    TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK-NEXT:    VFPtr {
; CHECK-NEXT:      TypeLeafKind: LF_VFUNCTAB (0x1409)
; CHECK-NEXT:      Type: <vftable 2 methods>* ([[vptr_2]])
; CHECK-NEXT:    }

; CHECK:        Struct ({{.*}}) {
; CHECK-NEXT:     TypeLeafKind: LF_STRUCTURE (0x1505)
; CHECK-NEXT:     MemberCount: 4
; CHECK-NEXT:     Properties [ (0x200)
; CHECK-NEXT:       HasUniqueName (0x200)
; CHECK-NEXT:     ]
; CHECK-NEXT:     FieldList: <field list> ([[b_members]])
; CHECK-NEXT:     DerivedFrom: 0x0
; CHECK-NEXT:     VShape: <vftable 2 methods> ([[vshape_2]])
; CHECK-NEXT:     SizeOf: 16
; CHECK-NEXT:     Name: B
; CHECK-NEXT:     LinkageName: .?AUB@@
; CHECK-NEXT:   }

; C has a primary base, so it does not need a VFPtr member.

; CHECK:       FieldList ([[c_members:0x[A-Z0-9]+]]) {
; CHECK-NEXT:    TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK-NEXT:    BaseClass {
; CHECK-NEXT:      TypeLeafKind: LF_BCLASS (0x1400)
; CHECK-NEXT:      AccessSpecifier: Public (0x3)
; CHECK-NEXT:      BaseType: A
; CHECK-NEXT:      BaseOffset: 0x0
; CHECK-NEXT:    }
; CHECK-NEXT:    BaseClass {
; CHECK-NEXT:      TypeLeafKind: LF_BCLASS (0x1400)
; CHECK-NEXT:      AccessSpecifier: Public (0x3)
; CHECK-NEXT:      BaseType: B
; CHECK-NEXT:      BaseOffset: 0x10
; CHECK-NEXT:    }
; CHECK-NOT:     VFPtr

; CHECK:        Struct ({{.*}}) {
; CHECK-NEXT:     TypeLeafKind: LF_STRUCTURE (0x1505)
; CHECK-NEXT:     MemberCount: 5
; CHECK-NEXT:     Properties [ (0x200)
; CHECK-NEXT:       HasUniqueName (0x200)
; CHECK-NEXT:     ]
; CHECK-NEXT:     FieldList: <field list> ([[c_members]])
; CHECK-NEXT:     DerivedFrom: 0x0
; CHECK-NEXT:     VShape: <vftable 1 methods> ([[vshape_1]])
; CHECK-NEXT:     SizeOf: 40
; CHECK-NEXT:     Name: C
; CHECK-NEXT:     LinkageName: .?AUC@@
; CHECK-NEXT:   }

; CHECK:       FieldList ([[d_members:0x[A-Z0-9]+]]) {
; CHECK-NEXT:    TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK-NEXT:    BaseClass {
; CHECK-NEXT:      TypeLeafKind: LF_BCLASS (0x1400)
; CHECK-NEXT:      AccessSpecifier: Public (0x3)
; CHECK-NEXT:      BaseType: C
; CHECK-NEXT:      BaseOffset: 0x0
; CHECK-NEXT:    }
; CHECK-NOT:     VFPtr

; CHECK:        Struct ({{.*}}) {
; CHECK-NEXT:     TypeLeafKind: LF_STRUCTURE (0x1505)
; CHECK-NEXT:     MemberCount: 4
; CHECK-NEXT:     Properties [ (0x200)
; CHECK-NEXT:       HasUniqueName (0x200)
; CHECK-NEXT:     ]
; CHECK-NEXT:     FieldList: <field list> ([[d_members]])
; CHECK-NEXT:     DerivedFrom: 0x0
; CHECK-NEXT:     VShape: <vftable 1 methods> ([[vshape_1]])
; CHECK-NEXT:     SizeOf: 48
; CHECK-NEXT:     Name: D
; CHECK-NEXT:     LinkageName: .?AUD@@
; CHECK-NEXT:   }

; ModuleID = 't.cpp'
source_filename = "t.cpp"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.0.24210"

%rtti.CompleteObjectLocator = type { i32, i32, i32, i32, i32, i32 }
%rtti.TypeDescriptor7 = type { i8**, i8*, [8 x i8] }
%rtti.ClassHierarchyDescriptor = type { i32, i32, i32, i32 }
%rtti.BaseClassDescriptor = type { i32, i32, i32, i32, i32, i32, i32 }
%struct.A = type { i32 (...)**, i32 }
%struct.B = type { i32 (...)**, i32 }
%struct.C = type { %struct.A, %struct.B, i32 }
%struct.D = type { %struct.C, i32 }

$"\01??0A@@QEAA@XZ" = comdat any

$"\01??0B@@QEAA@XZ" = comdat any

$"\01??0C@@QEAA@XZ" = comdat any

$"\01??0D@@QEAA@XZ" = comdat any

$"\01?g@C@@WBA@EAAXXZ" = comdat any

$"\01?g@D@@WBA@EAAXXZ" = comdat any

$"\01??_7A@@6B@" = comdat largest

$"\01??_R4A@@6B@" = comdat any

$"\01??_R0?AUA@@@8" = comdat any

$"\01??_R3A@@8" = comdat any

$"\01??_R2A@@8" = comdat any

$"\01??_R1A@?0A@EA@A@@8" = comdat any

$"\01??_7B@@6B@" = comdat largest

$"\01??_R4B@@6B@" = comdat any

$"\01??_R0?AUB@@@8" = comdat any

$"\01??_R3B@@8" = comdat any

$"\01??_R2B@@8" = comdat any

$"\01??_R1A@?0A@EA@B@@8" = comdat any

$"\01??_7C@@6BA@@@" = comdat largest

$"\01??_7C@@6BB@@@" = comdat largest

$"\01??_R4C@@6BA@@@" = comdat any

$"\01??_R0?AUC@@@8" = comdat any

$"\01??_R3C@@8" = comdat any

$"\01??_R2C@@8" = comdat any

$"\01??_R1A@?0A@EA@C@@8" = comdat any

$"\01??_R1BA@?0A@EA@B@@8" = comdat any

$"\01??_R4C@@6BB@@@" = comdat any

$"\01??_7D@@6BA@@@" = comdat largest

$"\01??_7D@@6BB@@@" = comdat largest

$"\01??_R4D@@6BA@@@" = comdat any

$"\01??_R0?AUD@@@8" = comdat any

$"\01??_R3D@@8" = comdat any

$"\01??_R2D@@8" = comdat any

$"\01??_R1A@?0A@EA@D@@8" = comdat any

$"\01??_R4D@@6BB@@@" = comdat any

@0 = private unnamed_addr constant [2 x i8*] [i8* bitcast (%rtti.CompleteObjectLocator* @"\01??_R4A@@6B@" to i8*), i8* bitcast (void (%struct.A*)* @"\01?g@A@@UEAAXXZ" to i8*)], comdat($"\01??_7A@@6B@")
@"\01??_R4A@@6B@" = linkonce_odr constant %rtti.CompleteObjectLocator { i32 1, i32 0, i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.TypeDescriptor7* @"\01??_R0?AUA@@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.ClassHierarchyDescriptor* @"\01??_R3A@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.CompleteObjectLocator* @"\01??_R4A@@6B@" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32) }, comdat
@"\01??_7type_info@@6B@" = external constant i8*
@"\01??_R0?AUA@@@8" = linkonce_odr global %rtti.TypeDescriptor7 { i8** @"\01??_7type_info@@6B@", i8* null, [8 x i8] c".?AUA@@\00" }, comdat
@__ImageBase = external constant i8
@"\01??_R3A@@8" = linkonce_odr constant %rtti.ClassHierarchyDescriptor { i32 0, i32 0, i32 1, i32 trunc (i64 sub nuw nsw (i64 ptrtoint ([2 x i32]* @"\01??_R2A@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32) }, comdat
@"\01??_R2A@@8" = linkonce_odr constant [2 x i32] [i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.BaseClassDescriptor* @"\01??_R1A@?0A@EA@A@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 0], comdat
@"\01??_R1A@?0A@EA@A@@8" = linkonce_odr constant %rtti.BaseClassDescriptor { i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.TypeDescriptor7* @"\01??_R0?AUA@@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 0, i32 0, i32 -1, i32 0, i32 64, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.ClassHierarchyDescriptor* @"\01??_R3A@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32) }, comdat
@1 = private unnamed_addr constant [3 x i8*] [i8* bitcast (%rtti.CompleteObjectLocator* @"\01??_R4B@@6B@" to i8*), i8* bitcast (void (%struct.B*)* @"\01?g@B@@UEAAXXZ" to i8*), i8* bitcast (void (%struct.B*)* @"\01?f@B@@UEAAXXZ" to i8*)], comdat($"\01??_7B@@6B@")
@"\01??_R4B@@6B@" = linkonce_odr constant %rtti.CompleteObjectLocator { i32 1, i32 0, i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.TypeDescriptor7* @"\01??_R0?AUB@@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.ClassHierarchyDescriptor* @"\01??_R3B@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.CompleteObjectLocator* @"\01??_R4B@@6B@" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32) }, comdat
@"\01??_R0?AUB@@@8" = linkonce_odr global %rtti.TypeDescriptor7 { i8** @"\01??_7type_info@@6B@", i8* null, [8 x i8] c".?AUB@@\00" }, comdat
@"\01??_R3B@@8" = linkonce_odr constant %rtti.ClassHierarchyDescriptor { i32 0, i32 0, i32 1, i32 trunc (i64 sub nuw nsw (i64 ptrtoint ([2 x i32]* @"\01??_R2B@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32) }, comdat
@"\01??_R2B@@8" = linkonce_odr constant [2 x i32] [i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.BaseClassDescriptor* @"\01??_R1A@?0A@EA@B@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 0], comdat
@"\01??_R1A@?0A@EA@B@@8" = linkonce_odr constant %rtti.BaseClassDescriptor { i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.TypeDescriptor7* @"\01??_R0?AUB@@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 0, i32 0, i32 -1, i32 0, i32 64, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.ClassHierarchyDescriptor* @"\01??_R3B@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32) }, comdat
@2 = private unnamed_addr constant [2 x i8*] [i8* bitcast (%rtti.CompleteObjectLocator* @"\01??_R4C@@6BA@@@" to i8*), i8* bitcast (void (%struct.C*)* @"\01?g@C@@UEAAXXZ" to i8*)], comdat($"\01??_7C@@6BA@@@")
@3 = private unnamed_addr constant [3 x i8*] [i8* bitcast (%rtti.CompleteObjectLocator* @"\01??_R4C@@6BB@@@" to i8*), i8* bitcast (void (%struct.C*)* @"\01?g@C@@WBA@EAAXXZ" to i8*), i8* bitcast (void (i8*)* @"\01?f@C@@UEAAXXZ" to i8*)], comdat($"\01??_7C@@6BB@@@")
@"\01??_R4C@@6BA@@@" = linkonce_odr constant %rtti.CompleteObjectLocator { i32 1, i32 0, i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.TypeDescriptor7* @"\01??_R0?AUC@@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.ClassHierarchyDescriptor* @"\01??_R3C@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.CompleteObjectLocator* @"\01??_R4C@@6BA@@@" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32) }, comdat
@"\01??_R0?AUC@@@8" = linkonce_odr global %rtti.TypeDescriptor7 { i8** @"\01??_7type_info@@6B@", i8* null, [8 x i8] c".?AUC@@\00" }, comdat
@"\01??_R3C@@8" = linkonce_odr constant %rtti.ClassHierarchyDescriptor { i32 0, i32 1, i32 3, i32 trunc (i64 sub nuw nsw (i64 ptrtoint ([4 x i32]* @"\01??_R2C@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32) }, comdat
@"\01??_R2C@@8" = linkonce_odr constant [4 x i32] [i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.BaseClassDescriptor* @"\01??_R1A@?0A@EA@C@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.BaseClassDescriptor* @"\01??_R1A@?0A@EA@A@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.BaseClassDescriptor* @"\01??_R1BA@?0A@EA@B@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 0], comdat
@"\01??_R1A@?0A@EA@C@@8" = linkonce_odr constant %rtti.BaseClassDescriptor { i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.TypeDescriptor7* @"\01??_R0?AUC@@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 2, i32 0, i32 -1, i32 0, i32 64, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.ClassHierarchyDescriptor* @"\01??_R3C@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32) }, comdat
@"\01??_R1BA@?0A@EA@B@@8" = linkonce_odr constant %rtti.BaseClassDescriptor { i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.TypeDescriptor7* @"\01??_R0?AUB@@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 0, i32 16, i32 -1, i32 0, i32 64, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.ClassHierarchyDescriptor* @"\01??_R3B@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32) }, comdat
@"\01??_R4C@@6BB@@@" = linkonce_odr constant %rtti.CompleteObjectLocator { i32 1, i32 16, i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.TypeDescriptor7* @"\01??_R0?AUC@@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.ClassHierarchyDescriptor* @"\01??_R3C@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.CompleteObjectLocator* @"\01??_R4C@@6BB@@@" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32) }, comdat
@4 = private unnamed_addr constant [2 x i8*] [i8* bitcast (%rtti.CompleteObjectLocator* @"\01??_R4D@@6BA@@@" to i8*), i8* bitcast (void (%struct.D*)* @"\01?g@D@@UEAAXXZ" to i8*)], comdat($"\01??_7D@@6BA@@@")
@5 = private unnamed_addr constant [3 x i8*] [i8* bitcast (%rtti.CompleteObjectLocator* @"\01??_R4D@@6BB@@@" to i8*), i8* bitcast (void (%struct.D*)* @"\01?g@D@@WBA@EAAXXZ" to i8*), i8* bitcast (void (i8*)* @"\01?f@D@@UEAAXXZ" to i8*)], comdat($"\01??_7D@@6BB@@@")
@"\01??_R4D@@6BA@@@" = linkonce_odr constant %rtti.CompleteObjectLocator { i32 1, i32 0, i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.TypeDescriptor7* @"\01??_R0?AUD@@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.ClassHierarchyDescriptor* @"\01??_R3D@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.CompleteObjectLocator* @"\01??_R4D@@6BA@@@" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32) }, comdat
@"\01??_R0?AUD@@@8" = linkonce_odr global %rtti.TypeDescriptor7 { i8** @"\01??_7type_info@@6B@", i8* null, [8 x i8] c".?AUD@@\00" }, comdat
@"\01??_R3D@@8" = linkonce_odr constant %rtti.ClassHierarchyDescriptor { i32 0, i32 1, i32 4, i32 trunc (i64 sub nuw nsw (i64 ptrtoint ([5 x i32]* @"\01??_R2D@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32) }, comdat
@"\01??_R2D@@8" = linkonce_odr constant [5 x i32] [i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.BaseClassDescriptor* @"\01??_R1A@?0A@EA@D@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.BaseClassDescriptor* @"\01??_R1A@?0A@EA@C@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.BaseClassDescriptor* @"\01??_R1A@?0A@EA@A@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.BaseClassDescriptor* @"\01??_R1BA@?0A@EA@B@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 0], comdat
@"\01??_R1A@?0A@EA@D@@8" = linkonce_odr constant %rtti.BaseClassDescriptor { i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.TypeDescriptor7* @"\01??_R0?AUD@@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 3, i32 0, i32 -1, i32 0, i32 64, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.ClassHierarchyDescriptor* @"\01??_R3D@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32) }, comdat
@"\01??_R4D@@6BB@@@" = linkonce_odr constant %rtti.CompleteObjectLocator { i32 1, i32 16, i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.TypeDescriptor7* @"\01??_R0?AUD@@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.ClassHierarchyDescriptor* @"\01??_R3D@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.CompleteObjectLocator* @"\01??_R4D@@6BB@@@" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32) }, comdat

@"\01??_7A@@6B@" = unnamed_addr alias i8*, getelementptr inbounds ([2 x i8*], [2 x i8*]* @0, i32 0, i32 1)
@"\01??_7B@@6B@" = unnamed_addr alias i8*, getelementptr inbounds ([3 x i8*], [3 x i8*]* @1, i32 0, i32 1)
@"\01??_7C@@6BA@@@" = unnamed_addr alias i8*, getelementptr inbounds ([2 x i8*], [2 x i8*]* @2, i32 0, i32 1)
@"\01??_7C@@6BB@@@" = unnamed_addr alias i8*, getelementptr inbounds ([3 x i8*], [3 x i8*]* @3, i32 0, i32 1)
@"\01??_7D@@6BA@@@" = unnamed_addr alias i8*, getelementptr inbounds ([2 x i8*], [2 x i8*]* @4, i32 0, i32 1)
@"\01??_7D@@6BB@@@" = unnamed_addr alias i8*, getelementptr inbounds ([3 x i8*], [3 x i8*]* @5, i32 0, i32 1)

; Function Attrs: nounwind uwtable
define void @"\01?h@@YAXXZ"() #0 !dbg !7 {
entry:
  %a = alloca %struct.A, align 8
  %b = alloca %struct.B, align 8
  %c = alloca %struct.C, align 8
  %d = alloca %struct.D, align 8
  call void @llvm.dbg.declare(metadata %struct.A* %a, metadata !10, metadata !22), !dbg !23
  %call = call %struct.A* @"\01??0A@@QEAA@XZ"(%struct.A* %a) #5, !dbg !23
  call void @llvm.dbg.declare(metadata %struct.B* %b, metadata !24, metadata !22), !dbg !36
  %call1 = call %struct.B* @"\01??0B@@QEAA@XZ"(%struct.B* %b) #5, !dbg !36
  call void @llvm.dbg.declare(metadata %struct.C* %c, metadata !37, metadata !22), !dbg !48
  %call2 = call %struct.C* @"\01??0C@@QEAA@XZ"(%struct.C* %c) #5, !dbg !48
  call void @llvm.dbg.declare(metadata %struct.D* %d, metadata !49, metadata !22), !dbg !59
  %call3 = call %struct.D* @"\01??0D@@QEAA@XZ"(%struct.D* %d) #5, !dbg !59
  ret void, !dbg !60
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr %struct.A* @"\01??0A@@QEAA@XZ"(%struct.A* returned %this) unnamed_addr #2 comdat align 2 !dbg !61 {
entry:
  %this.addr = alloca %struct.A*, align 8
  store %struct.A* %this, %struct.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.A** %this.addr, metadata !63, metadata !22), !dbg !65
  %this1 = load %struct.A*, %struct.A** %this.addr, align 8
  %0 = bitcast %struct.A* %this1 to i32 (...)***, !dbg !66
  store i32 (...)** bitcast (i8** @"\01??_7A@@6B@" to i32 (...)**), i32 (...)*** %0, align 8, !dbg !66
  %a = getelementptr inbounds %struct.A, %struct.A* %this1, i32 0, i32 1, !dbg !67
  store i32 0, i32* %a, align 8, !dbg !67
  ret %struct.A* %this1, !dbg !66
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr %struct.B* @"\01??0B@@QEAA@XZ"(%struct.B* returned %this) unnamed_addr #2 comdat align 2 !dbg !68 {
entry:
  %this.addr = alloca %struct.B*, align 8
  store %struct.B* %this, %struct.B** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.B** %this.addr, metadata !70, metadata !22), !dbg !72
  %this1 = load %struct.B*, %struct.B** %this.addr, align 8
  %0 = bitcast %struct.B* %this1 to i32 (...)***, !dbg !73
  store i32 (...)** bitcast (i8** @"\01??_7B@@6B@" to i32 (...)**), i32 (...)*** %0, align 8, !dbg !73
  %b = getelementptr inbounds %struct.B, %struct.B* %this1, i32 0, i32 1, !dbg !74
  store i32 0, i32* %b, align 8, !dbg !74
  ret %struct.B* %this1, !dbg !73
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr %struct.C* @"\01??0C@@QEAA@XZ"(%struct.C* returned %this) unnamed_addr #2 comdat align 2 !dbg !75 {
entry:
  %this.addr = alloca %struct.C*, align 8
  store %struct.C* %this, %struct.C** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.C** %this.addr, metadata !77, metadata !22), !dbg !79
  %this1 = load %struct.C*, %struct.C** %this.addr, align 8
  %0 = bitcast %struct.C* %this1 to %struct.A*, !dbg !80
  %call = call %struct.A* @"\01??0A@@QEAA@XZ"(%struct.A* %0) #5, !dbg !80
  %1 = bitcast %struct.C* %this1 to i8*, !dbg !80
  %2 = getelementptr inbounds i8, i8* %1, i64 16, !dbg !80
  %3 = bitcast i8* %2 to %struct.B*, !dbg !80
  %call2 = call %struct.B* @"\01??0B@@QEAA@XZ"(%struct.B* %3) #5, !dbg !80
  %4 = bitcast %struct.C* %this1 to i32 (...)***, !dbg !80
  store i32 (...)** bitcast (i8** @"\01??_7C@@6BA@@@" to i32 (...)**), i32 (...)*** %4, align 8, !dbg !80
  %5 = bitcast %struct.C* %this1 to i8*, !dbg !80
  %add.ptr = getelementptr inbounds i8, i8* %5, i64 16, !dbg !80
  %6 = bitcast i8* %add.ptr to i32 (...)***, !dbg !80
  store i32 (...)** bitcast (i8** @"\01??_7C@@6BB@@@" to i32 (...)**), i32 (...)*** %6, align 8, !dbg !80
  %c = getelementptr inbounds %struct.C, %struct.C* %this1, i32 0, i32 2, !dbg !81
  store i32 0, i32* %c, align 8, !dbg !81
  ret %struct.C* %this1, !dbg !80
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr %struct.D* @"\01??0D@@QEAA@XZ"(%struct.D* returned %this) unnamed_addr #2 comdat align 2 !dbg !82 {
entry:
  %this.addr = alloca %struct.D*, align 8
  store %struct.D* %this, %struct.D** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.D** %this.addr, metadata !84, metadata !22), !dbg !86
  %this1 = load %struct.D*, %struct.D** %this.addr, align 8
  %0 = bitcast %struct.D* %this1 to %struct.C*, !dbg !87
  %call = call %struct.C* @"\01??0C@@QEAA@XZ"(%struct.C* %0) #5, !dbg !87
  %1 = bitcast %struct.D* %this1 to i32 (...)***, !dbg !87
  store i32 (...)** bitcast (i8** @"\01??_7D@@6BA@@@" to i32 (...)**), i32 (...)*** %1, align 8, !dbg !87
  %2 = bitcast %struct.D* %this1 to i8*, !dbg !87
  %add.ptr = getelementptr inbounds i8, i8* %2, i64 16, !dbg !87
  %3 = bitcast i8* %add.ptr to i32 (...)***, !dbg !87
  store i32 (...)** bitcast (i8** @"\01??_7D@@6BB@@@" to i32 (...)**), i32 (...)*** %3, align 8, !dbg !87
  %d = getelementptr inbounds %struct.D, %struct.D* %this1, i32 0, i32 1, !dbg !88
  store i32 0, i32* %d, align 8, !dbg !88
  ret %struct.D* %this1, !dbg !87
}

declare void @"\01?g@A@@UEAAXXZ"(%struct.A*) unnamed_addr #3

declare void @"\01?g@B@@UEAAXXZ"(%struct.B*) unnamed_addr #3

declare void @"\01?f@B@@UEAAXXZ"(%struct.B*) unnamed_addr #3

declare void @"\01?g@C@@UEAAXXZ"(%struct.C*) unnamed_addr #3

; Function Attrs: uwtable
define linkonce_odr void @"\01?g@C@@WBA@EAAXXZ"(%struct.C* %this) unnamed_addr #4 comdat align 2 !dbg !89 {
entry:
  %this.addr = alloca %struct.C*, align 8
  store %struct.C* %this, %struct.C** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.C** %this.addr, metadata !91, metadata !22), !dbg !92
  %this1 = load %struct.C*, %struct.C** %this.addr, align 8, !dbg !93
  %0 = bitcast %struct.C* %this1 to i8*, !dbg !93
  %1 = getelementptr i8, i8* %0, i32 -16, !dbg !93
  %2 = bitcast i8* %1 to %struct.C*, !dbg !93
  tail call void @"\01?g@C@@UEAAXXZ"(%struct.C* %2), !dbg !93
  ret void, !dbg !93
}

declare void @"\01?f@C@@UEAAXXZ"(i8*) unnamed_addr #3

declare void @"\01?g@D@@UEAAXXZ"(%struct.D*) unnamed_addr #3

; Function Attrs: uwtable
define linkonce_odr void @"\01?g@D@@WBA@EAAXXZ"(%struct.D* %this) unnamed_addr #4 comdat align 2 !dbg !94 {
entry:
  %this.addr = alloca %struct.D*, align 8
  store %struct.D* %this, %struct.D** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.D** %this.addr, metadata !95, metadata !22), !dbg !96
  %this1 = load %struct.D*, %struct.D** %this.addr, align 8, !dbg !97
  %0 = bitcast %struct.D* %this1 to i8*, !dbg !97
  %1 = getelementptr i8, i8* %0, i32 -16, !dbg !97
  %2 = bitcast i8* %1 to %struct.D*, !dbg !97
  tail call void @"\01?g@D@@UEAAXXZ"(%struct.D* %2), !dbg !97
  ret void, !dbg !97
}

declare void @"\01?f@D@@UEAAXXZ"(i8*) unnamed_addr #3

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { inlinehint nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 4.0.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "t.cpp", directory: "C:\5Csrc\5Cllvm\5Cbuild")
!2 = !{}
!3 = !{i32 2, !"CodeView", i32 1}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"PIC Level", i32 2}
!6 = !{!"clang version 4.0.0 "}
!7 = distinct !DISubprogram(name: "h", linkageName: "\01?h@@YAXXZ", scope: !1, file: !1, line: 20, type: !8, isLocal: false, isDefinition: true, scopeLine: 20, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !DILocalVariable(name: "a", scope: !7, file: !1, line: 22, type: !11)
!11 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "A", file: !1, line: 1, size: 128, align: 64, elements: !12, vtableHolder: !11, identifier: ".?AUA@@")
!12 = !{!13, !14, !16, !18}
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "__vtbl_ptr_type", baseType: null, size: 64)
!14 = !DIDerivedType(tag: DW_TAG_member, name: "_vptr$A", scope: !1, file: !1, baseType: !15, size: 64, flags: DIFlagArtificial)
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64)
!16 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !11, file: !1, line: 3, baseType: !17, size: 32, align: 32, offset: 64)
!17 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!18 = !DISubprogram(name: "g", linkageName: "\01?g@A@@UEAAXXZ", scope: !11, file: !1, line: 2, type: !19, isLocal: false, isDefinition: false, scopeLine: 2, containingType: !11, virtuality: DW_VIRTUALITY_virtual, virtualIndex: 0, flags: DIFlagPrototyped | DIFlagIntroducedVirtual, isOptimized: false)
!19 = !DISubroutineType(types: !20)
!20 = !{null, !21}
!21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!22 = !DIExpression()
!23 = !DILocation(line: 22, column: 5, scope: !7)
!24 = !DILocalVariable(name: "b", scope: !7, file: !1, line: 23, type: !25)
!25 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "B", file: !1, line: 5, size: 128, align: 64, elements: !26, vtableHolder: !25, identifier: ".?AUB@@")
!26 = !{!27, !28, !30, !31, !35}
!27 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "__vtbl_ptr_type", baseType: null, size: 128)
!28 = !DIDerivedType(tag: DW_TAG_member, name: "_vptr$B", scope: !1, file: !1, baseType: !29, size: 64, flags: DIFlagArtificial)
!29 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !27, size: 64)
!30 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !25, file: !1, line: 8, baseType: !17, size: 32, align: 32, offset: 64)
!31 = !DISubprogram(name: "g", linkageName: "\01?g@B@@UEAAXXZ", scope: !25, file: !1, line: 6, type: !32, isLocal: false, isDefinition: false, scopeLine: 6, containingType: !25, virtuality: DW_VIRTUALITY_virtual, virtualIndex: 0, flags: DIFlagPrototyped | DIFlagIntroducedVirtual, isOptimized: false)
!32 = !DISubroutineType(types: !33)
!33 = !{null, !34}
!34 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !25, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!35 = !DISubprogram(name: "f", linkageName: "\01?f@B@@UEAAXXZ", scope: !25, file: !1, line: 7, type: !32, isLocal: false, isDefinition: false, scopeLine: 7, containingType: !25, virtuality: DW_VIRTUALITY_virtual, virtualIndex: 1, flags: DIFlagPrototyped | DIFlagIntroducedVirtual, isOptimized: false)
!36 = !DILocation(line: 23, column: 5, scope: !7)
!37 = !DILocalVariable(name: "c", scope: !7, file: !1, line: 24, type: !38)
!38 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "C", file: !1, line: 10, size: 320, align: 64, elements: !39, vtableHolder: !11, identifier: ".?AUC@@")
!39 = !{!40, !41, !13, !42, !43, !47}
!40 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !38, baseType: !11)
!41 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !38, baseType: !25, offset: 128)
!42 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !38, file: !1, line: 13, baseType: !17, size: 32, align: 32, offset: 256)
!43 = !DISubprogram(name: "g", linkageName: "\01?g@C@@UEAAXXZ", scope: !38, file: !1, line: 11, type: !44, isLocal: false, isDefinition: false, scopeLine: 11, containingType: !38, virtuality: DW_VIRTUALITY_virtual, virtualIndex: 0, flags: DIFlagPrototyped, isOptimized: false)
!44 = !DISubroutineType(types: !45)
!45 = !{null, !46}
!46 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !38, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!47 = !DISubprogram(name: "f", linkageName: "\01?f@C@@UEAAXXZ", scope: !38, file: !1, line: 12, type: !44, isLocal: false, isDefinition: false, scopeLine: 12, containingType: !38, virtuality: DW_VIRTUALITY_virtual, virtualIndex: 1, thisAdjustment: 16, flags: DIFlagPrototyped, isOptimized: false)
!48 = !DILocation(line: 24, column: 5, scope: !7)
!49 = !DILocalVariable(name: "d", scope: !7, file: !1, line: 25, type: !50)
!50 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "D", file: !1, line: 15, size: 384, align: 64, elements: !51, vtableHolder: !11, identifier: ".?AUD@@")
!51 = !{!52, !13, !53, !54, !58}
!52 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !50, baseType: !38)
!53 = !DIDerivedType(tag: DW_TAG_member, name: "d", scope: !50, file: !1, line: 18, baseType: !17, size: 32, align: 32, offset: 320)
!54 = !DISubprogram(name: "g", linkageName: "\01?g@D@@UEAAXXZ", scope: !50, file: !1, line: 16, type: !55, isLocal: false, isDefinition: false, scopeLine: 16, containingType: !50, virtuality: DW_VIRTUALITY_virtual, virtualIndex: 0, flags: DIFlagPrototyped, isOptimized: false)
!55 = !DISubroutineType(types: !56)
!56 = !{null, !57}
!57 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !50, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!58 = !DISubprogram(name: "f", linkageName: "\01?f@D@@UEAAXXZ", scope: !50, file: !1, line: 17, type: !55, isLocal: false, isDefinition: false, scopeLine: 17, containingType: !50, virtuality: DW_VIRTUALITY_virtual, virtualIndex: 1, thisAdjustment: 16, flags: DIFlagPrototyped, isOptimized: false)
!59 = !DILocation(line: 25, column: 5, scope: !7)
!60 = !DILocation(line: 26, column: 1, scope: !7)
!61 = distinct !DISubprogram(name: "A", linkageName: "\01??0A@@QEAA@XZ", scope: !11, file: !1, line: 1, type: !19, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false, unit: !0, declaration: !62, retainedNodes: !2)
!62 = !DISubprogram(name: "A", scope: !11, type: !19, isLocal: false, isDefinition: false, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false)
!63 = !DILocalVariable(name: "this", arg: 1, scope: !61, type: !64, flags: DIFlagArtificial | DIFlagObjectPointer)
!64 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 64, align: 64)
!65 = !DILocation(line: 0, scope: !61)
!66 = !DILocation(line: 1, column: 8, scope: !61)
!67 = !DILocation(line: 3, column: 7, scope: !61)
!68 = distinct !DISubprogram(name: "B", linkageName: "\01??0B@@QEAA@XZ", scope: !25, file: !1, line: 5, type: !32, isLocal: false, isDefinition: true, scopeLine: 5, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false, unit: !0, declaration: !69, retainedNodes: !2)
!69 = !DISubprogram(name: "B", scope: !25, type: !32, isLocal: false, isDefinition: false, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false)
!70 = !DILocalVariable(name: "this", arg: 1, scope: !68, type: !71, flags: DIFlagArtificial | DIFlagObjectPointer)
!71 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !25, size: 64, align: 64)
!72 = !DILocation(line: 0, scope: !68)
!73 = !DILocation(line: 5, column: 8, scope: !68)
!74 = !DILocation(line: 8, column: 7, scope: !68)
!75 = distinct !DISubprogram(name: "C", linkageName: "\01??0C@@QEAA@XZ", scope: !38, file: !1, line: 10, type: !44, isLocal: false, isDefinition: true, scopeLine: 10, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false, unit: !0, declaration: !76, retainedNodes: !2)
!76 = !DISubprogram(name: "C", scope: !38, type: !44, isLocal: false, isDefinition: false, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false)
!77 = !DILocalVariable(name: "this", arg: 1, scope: !75, type: !78, flags: DIFlagArtificial | DIFlagObjectPointer)
!78 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !38, size: 64, align: 64)
!79 = !DILocation(line: 0, scope: !75)
!80 = !DILocation(line: 10, column: 8, scope: !75)
!81 = !DILocation(line: 13, column: 7, scope: !75)
!82 = distinct !DISubprogram(name: "D", linkageName: "\01??0D@@QEAA@XZ", scope: !50, file: !1, line: 15, type: !55, isLocal: false, isDefinition: true, scopeLine: 15, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false, unit: !0, declaration: !83, retainedNodes: !2)
!83 = !DISubprogram(name: "D", scope: !50, type: !55, isLocal: false, isDefinition: false, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false)
!84 = !DILocalVariable(name: "this", arg: 1, scope: !82, type: !85, flags: DIFlagArtificial | DIFlagObjectPointer)
!85 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !50, size: 64, align: 64)
!86 = !DILocation(line: 0, scope: !82)
!87 = !DILocation(line: 15, column: 8, scope: !82)
!88 = !DILocation(line: 18, column: 7, scope: !82)
!89 = distinct !DISubprogram(linkageName: "\01?g@C@@WBA@EAAXXZ", scope: !1, file: !1, line: 11, type: !90, isLocal: false, isDefinition: true, scopeLine: 11, flags: DIFlagArtificial, isOptimized: false, unit: !0, retainedNodes: !2)
!90 = !DISubroutineType(types: !2)
!91 = !DILocalVariable(name: "this", arg: 1, scope: !89, type: !78, flags: DIFlagArtificial | DIFlagObjectPointer)
!92 = !DILocation(line: 0, scope: !89)
!93 = !DILocation(line: 11, column: 16, scope: !89)
!94 = distinct !DISubprogram(linkageName: "\01?g@D@@WBA@EAAXXZ", scope: !1, file: !1, line: 16, type: !90, isLocal: false, isDefinition: true, scopeLine: 16, flags: DIFlagArtificial, isOptimized: false, unit: !0, retainedNodes: !2)
!95 = !DILocalVariable(name: "this", arg: 1, scope: !94, type: !85, flags: DIFlagArtificial | DIFlagObjectPointer)
!96 = !DILocation(line: 0, scope: !94)
!97 = !DILocation(line: 16, column: 16, scope: !94)
