; RUN: llc < %s -filetype=obj | llvm-readobj - --codeview | FileCheck %s
; RUN: llc < %s | llvm-mc -filetype=obj --triple=x86_64-windows | llvm-readobj - --codeview | FileCheck %s
;
; Command to generate function-options.ll
; $ clang++ function-options.cpp -S -emit-llvm -g -gcodeview -o function-options.ll
;
; #define DEFINE_FUNCTION(T) \
;   T Func_##T(T &arg) { return arg; }
;
; class AClass {};
; DEFINE_FUNCTION(AClass); // Expect: FO = None
;
; class BClass {
; private:
;   explicit BClass(); // Expect ctor: FO = Constructor
; };
; DEFINE_FUNCTION(BClass); // Expect: FO = CxxReturnUdt
;
; class C1Class {
; public:
;   C1Class() = default; // Note: Clang generates one defaulted ctor (FO = None) while MSVC doesn't.
; };
; DEFINE_FUNCTION(C1Class); // Expect: FO = None
;
; class C2Class { // Note: MSVC-specific dtor, i.e. __vecDelDtor won't be verified in this case.
; public:
;   ~C2Class() {} // Expect ~C2Class: FO = None
; };
; DEFINE_FUNCTION(C2Class); // Expect: FO = CxxReturnUdt
;
; class DClass : public BClass {}; // Note: MSVC yields one compiler-generated ctor for DClass while clang doesn't.
; DEFINE_FUNCTION(DClass); // Expect: FO = CxxReturnUdt
;
; class FClass {
;   static int x;
;   AClass Member_AClass(AClass &);
;   BClass Member_BClass(BClass &);
; };
; DEFINE_FUNCTION(FClass); // Expect FO = None
; 
; struct AStruct {};
; DEFINE_FUNCTION(AStruct); // Expect FO = None
;
; struct BStruct { BStruct(); }; // Expect ctor: FO = Constructor
; DEFINE_FUNCTION(BStruct); // Expect FO = CxxReturnUdt
;
; union AUnion {};
; DEFINE_FUNCTION(AUnion); // Expect FO = None
;
; union BUnion { BUnion() = default; }; // Note: Clang generates one defaulted ctor (FO = None) while MSVC does not.
; DEFINE_FUNCTION(BUnion); // Expect FO = None


; CHECK: Format: COFF-x86-64
; CHECK: Arch: x86_64
; CHECK: AddressSize: 64bit
; CHECK: CodeViewTypes [
; CHECK:   Section: .debug$T ({{.*}})
; CHECK:   Magic: 0x4
; CHECK:   Procedure ([[SP_A:.*]]) {
; CHECK:     TypeLeafKind: LF_PROCEDURE (0x1008)
; CHECK:     ReturnType: AClass ({{.*}})
; CHECK:     CallingConvention: NearC (0x0)
; CHECK:     FunctionOptions [ (0x0)
; CHECK:     ]
; CHECK:     NumParameters: 1
; CHECK:     ArgListType: (AClass&) ({{.*}})
; CHECK:   }
; CHECK:   FuncId ({{.*}}) {
; CHECK:     TypeLeafKind: LF_FUNC_ID (0x1601)
; CHECK:     ParentScope: 0x0
; CHECK:     FunctionType: AClass (AClass&) ([[SP_A]])
; CHECK:     Name: Func_AClass
; CHECK:   }
; CHECK:   Procedure ([[SP_B:.*]]) {
; CHECK:     TypeLeafKind: LF_PROCEDURE (0x1008)
; CHECK:     ReturnType: BClass ({{.*}})
; CHECK:     CallingConvention: NearC (0x0)
; CHECK:     FunctionOptions [ (0x1)
; CHECK:       CxxReturnUdt (0x1)
; CHECK:     ]
; CHECK:     NumParameters: 1
; CHECK:     ArgListType: (BClass&) ({{.*}})
; CHECK:   }
; CHECK:   MemberFunction ([[CTOR_B:.*]]) {
; CHECK:     TypeLeafKind: LF_MFUNCTION (0x1009)
; CHECK:     ReturnType: void (0x3)
; CHECK:     ClassType: BClass ({{.*}})
; CHECK:     ThisType: BClass* {{.*}}
; CHECK:     CallingConvention: NearC (0x0)
; CHECK:     FunctionOptions [ (0x2)
; CHECK:       Constructor (0x2)
; CHECK:     ]
; CHECK:     NumParameters: 0
; CHECK:     ArgListType: () ({{.*}})
; CHECK:     ThisAdjustment: 0
; CHECK:   }
; CHECK:   FieldList ({{.*}}) {
; CHECK:     TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK:     OneMethod {
; CHECK:       TypeLeafKind: LF_ONEMETHOD (0x1511)
; CHECK:       AccessSpecifier: Private (0x1)
; CHECK:       Type: void BClass::() ([[CTOR_B]])
; CHECK:       Name: BClass
; CHECK:     }
; CHECK:   }
; CHECK:   FuncId ({{.*}}) {
; CHECK:     TypeLeafKind: LF_FUNC_ID (0x1601)
; CHECK:     ParentScope: 0x0
; CHECK:     FunctionType: BClass (BClass&) ([[SP_B]])
; CHECK:     Name: Func_BClass
; CHECK:   }
; CHECK:   Procedure ([[SP_C1:.*]]) {
; CHECK:     TypeLeafKind: LF_PROCEDURE (0x1008)
; CHECK:     ReturnType: C1Class ({{.*}})
; CHECK:     CallingConvention: NearC (0x0)
; CHECK:     FunctionOptions [ (0x0)
; CHECK:     ]
; CHECK:     NumParameters: 1
; CHECK:     ArgListType: (C1Class&) ({{.*}})
; CHECK:   }
; CHECK:   MemberFunction ([[CTOR_C1:.*]]) {
; CHECK:     TypeLeafKind: LF_MFUNCTION (0x1009)
; CHECK:     ReturnType: void (0x3)
; CHECK:     ClassType: C1Class ({{.*}})
; CHECK:     ThisType: C1Class* {{.*}}
; CHECK:     CallingConvention: NearC (0x0)
; CHECK:     FunctionOptions [ (0x0)
; CHECK:     ]
; CHECK:     NumParameters: 0
; CHECK:     ArgListType: () ({{.*}})
; CHECK:     ThisAdjustment: 0
; CHECK:   }
; CHECK:   FieldList ({{.*}}) {
; CHECK:     TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK:     OneMethod {
; CHECK:       TypeLeafKind: LF_ONEMETHOD (0x1511)
; CHECK:       AccessSpecifier: Public (0x3)
; CHECK:       Type: void C1Class::() ([[CTOR_C1]])
; CHECK:       Name: C1Class
; CHECK:     }
; CHECK:   }
; CHECK:   FuncId ({{.*}}) {
; CHECK:     TypeLeafKind: LF_FUNC_ID (0x1601)
; CHECK:     ParentScope: 0x0
; CHECK:     FunctionType: C1Class (C1Class&) ([[SP_C1]])
; CHECK:     Name: Func_C1Class
; CHECK:   }
; CHECK:   Procedure ([[SP_C2:.*]]) {
; CHECK:     TypeLeafKind: LF_PROCEDURE (0x1008)
; CHECK:     ReturnType: C2Class ({{.*}})
; CHECK:     CallingConvention: NearC (0x0)
; CHECK:     FunctionOptions [ (0x1)
; CHECK:       CxxReturnUdt (0x1)
; CHECK:     ]
; CHECK:     NumParameters: 1
; CHECK:     ArgListType: (C2Class&) ({{.*}})
; CHECK:   }
; CHECK:   MemberFunction ([[CTOR_C2:.*]]) {
; CHECK:     TypeLeafKind: LF_MFUNCTION (0x1009)
; CHECK:     ReturnType: void (0x3)
; CHECK:     ClassType: C2Class ({{.*}})
; CHECK:     ThisType: C2Class* {{.*}}
; CHECK:     CallingConvention: NearC (0x0)
; CHECK:     FunctionOptions [ (0x0)
; CHECK:     ]
; CHECK:     NumParameters: 0
; CHECK:     ArgListType: () ({{.*}})
; CHECK:     ThisAdjustment: 0
; CHECK:   }
; CHECK:   FieldList ({{.*}}) {
; CHECK:     TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK:     OneMethod {
; CHECK:       TypeLeafKind: LF_ONEMETHOD (0x1511)
; CHECK:       AccessSpecifier: Public (0x3)
; CHECK:       Type: void C2Class::() ([[CTOR_C2]])
; CHECK:       Name: ~C2Class
; CHECK:     }
; CHECK:   }
; CHECK:   FuncId ({{.*}}) {
; CHECK:     TypeLeafKind: LF_FUNC_ID (0x1601)
; CHECK:     ParentScope: 0x0
; CHECK:     FunctionType: C2Class (C2Class&) ([[SP_C2]])
; CHECK:     Name: Func_C2Class
; CHECK:   }
; CHECK:   Procedure ([[SP_D:.*]]) {
; CHECK:     TypeLeafKind: LF_PROCEDURE (0x1008)
; CHECK:     ReturnType: DClass ({{.*}})
; CHECK:     CallingConvention: NearC (0x0)
; CHECK:     FunctionOptions [ (0x1)
; CHECK:       CxxReturnUdt (0x1)
; CHECK:     ]
; CHECK:     NumParameters: 1
; CHECK:     ArgListType: (DClass&) ({{.*}})
; CHECK:   }
; CHECK:   FieldList ({{.*}}) {
; CHECK:     TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK:     BaseClass {
; CHECK:       TypeLeafKind: LF_BCLASS (0x1400)
; CHECK:       AccessSpecifier: Public (0x3)
; CHECK:       BaseType: BClass ({{.*}})
; CHECK:       BaseOffset: 0x0
; CHECK:     }
; CHECK:   }
; CHECK:   FuncId ({{.*}}) {
; CHECK:     TypeLeafKind: LF_FUNC_ID (0x1601)
; CHECK:     ParentScope: 0x0
; CHECK:     FunctionType: DClass (DClass&) ([[SP_D]])
; CHECK:     Name: Func_DClass
; CHECK:   }
; CHECK:   Procedure ([[SP_F:.*]]) {
; CHECK:     TypeLeafKind: LF_PROCEDURE (0x1008)
; CHECK:     ReturnType: FClass ({{.*}})
; CHECK:     CallingConvention: NearC (0x0)
; CHECK:     FunctionOptions [ (0x0)
; CHECK:     ]
; CHECK:     NumParameters: 1
; CHECK:     ArgListType: (FClass&) ({{.*}})
; CHECK:   }
; CHECK:   MemberFunction ([[MF_A:.*]]) {
; CHECK:     TypeLeafKind: LF_MFUNCTION (0x1009)
; CHECK:     ReturnType: AClass ({{.*}})
; CHECK:     ClassType: FClass ({{.*}})
; CHECK:     ThisType: FClass* {{.*}}
; CHECK:     CallingConvention: NearC (0x0)
; CHECK:     FunctionOptions [ (0x1)
; CHECK:       CxxReturnUdt (0x1)
; CHECK:     ]
; CHECK:     NumParameters: 1
; CHECK:     ArgListType: (AClass&) ({{.*}})
; CHECK:     ThisAdjustment: 0
; CHECK:   }
; CHECK:   MemberFunction ([[MF_B:.*]]) {
; CHECK:     TypeLeafKind: LF_MFUNCTION (0x1009)
; CHECK:     ReturnType: BClass ({{.*}})
; CHECK:     ClassType: FClass ({{.*}})
; CHECK:     ThisType: FClass* {{.*}}
; CHECK:     CallingConvention: NearC (0x0)
; CHECK:     FunctionOptions [ (0x1)
; CHECK:       CxxReturnUdt (0x1)
; CHECK:     ]
; CHECK:     NumParameters: 1
; CHECK:     ArgListType: (BClass&) ({{.*}})
; CHECK:     ThisAdjustment: 0
; CHECK:   }
; CHECK:   FieldList ({{.*}}) {
; CHECK:     TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK:     StaticDataMember {
; CHECK:       TypeLeafKind: LF_STMEMBER (0x150E)
; CHECK:       AccessSpecifier: Private (0x1)
; CHECK:       Type: int (0x74)
; CHECK:       Name: x
; CHECK:     }
; CHECK:     OneMethod {
; CHECK:       TypeLeafKind: LF_ONEMETHOD (0x1511)
; CHECK:       AccessSpecifier: Private (0x1)
; CHECK:       Type: AClass FClass::(AClass&) ([[MF_A]])
; CHECK:       Name: Member_AClass
; CHECK:     }
; CHECK:     OneMethod {
; CHECK:       TypeLeafKind: LF_ONEMETHOD (0x1511)
; CHECK:       AccessSpecifier: Private (0x1)
; CHECK:       Type: BClass FClass::(BClass&) ([[MF_B]])
; CHECK:       Name: Member_BClass
; CHECK:     }
; CHECK:   }
; CHECK:   FuncId ({{.*}}) {
; CHECK:     TypeLeafKind: LF_FUNC_ID (0x1601)
; CHECK:     ParentScope: 0x0
; CHECK:     FunctionType: FClass (FClass&) ([[SP_F]])
; CHECK:     Name: Func_FClass
; CHECK:   }
; CHECK:   Procedure ([[SP_AS:.*]]) {
; CHECK:     TypeLeafKind: LF_PROCEDURE (0x1008)
; CHECK:     ReturnType: AStruct ({{.*}})
; CHECK:     CallingConvention: NearC (0x0)
; CHECK:     FunctionOptions [ (0x0)
; CHECK:     ]
; CHECK:     NumParameters: 1
; CHECK:     ArgListType: (AStruct&) ({{.*}})
; CHECK:   }
; CHECK:   FuncId ({{.*}}) {
; CHECK:     TypeLeafKind: LF_FUNC_ID (0x1601)
; CHECK:     ParentScope: 0x0
; CHECK:     FunctionType: AStruct (AStruct&) ([[SP_AS]])
; CHECK:     Name: Func_AStruct
; CHECK:   }
; CHECK:   Procedure ([[SP_BS:.*]]) {
; CHECK:     TypeLeafKind: LF_PROCEDURE (0x1008)
; CHECK:     ReturnType: BStruct ({{.*}})
; CHECK:     CallingConvention: NearC (0x0)
; CHECK:     FunctionOptions [ (0x1)
; CHECK:       CxxReturnUdt (0x1)
; CHECK:     ]
; CHECK:     NumParameters: 1
; CHECK:     ArgListType: (BStruct&) ({{.*}})
; CHECK:   }
; CHECK:   MemberFunction ([[CTOR_BS:.*]]) {
; CHECK:     TypeLeafKind: LF_MFUNCTION (0x1009)
; CHECK:     ReturnType: void (0x3)
; CHECK:     ClassType: BStruct ({{.*}})
; CHECK:     ThisType: BStruct* {{.*}}
; CHECK:     CallingConvention: NearC (0x0)
; CHECK:     FunctionOptions [ (0x2)
; CHECK:       Constructor (0x2)
; CHECK:     ]
; CHECK:     NumParameters: 0
; CHECK:     ArgListType: () ({{.*}})
; CHECK:     ThisAdjustment: 0
; CHECK:   }
; CHECK:   FieldList ({{.*}}) {
; CHECK:     TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK:     OneMethod {
; CHECK:       TypeLeafKind: LF_ONEMETHOD (0x1511)
; CHECK:       AccessSpecifier: Public (0x3)
; CHECK:       Type: void BStruct::() ([[CTOR_BS]])
; CHECK:       Name: BStruct
; CHECK:     }
; CHECK:   }
; CHECK:   FuncId ({{.*}}) {
; CHECK:     TypeLeafKind: LF_FUNC_ID (0x1601)
; CHECK:     ParentScope: 0x0
; CHECK:     FunctionType: BStruct (BStruct&) ([[SP_BS]])
; CHECK:     Name: Func_BStruct
; CHECK:   }
; CHECK:   Procedure ([[SP_AU:.*]]) {
; CHECK:     TypeLeafKind: LF_PROCEDURE (0x1008)
; CHECK:     ReturnType: AUnion ({{.*}})
; CHECK:     CallingConvention: NearC (0x0)
; CHECK:     FunctionOptions [ (0x0)
; CHECK:     ]
; CHECK:     NumParameters: 1
; CHECK:     ArgListType: (AUnion&) ({{.*}})
; CHECK:   }
; CHECK:   FuncId ({{.*}}) {
; CHECK:     TypeLeafKind: LF_FUNC_ID (0x1601)
; CHECK:     ParentScope: 0x0
; CHECK:     FunctionType: AUnion (AUnion&) ([[SP_AU]])
; CHECK:     Name: Func_AUnion
; CHECK:   }
; CHECK:   Procedure ([[SP_BU:.*]]) {
; CHECK:     TypeLeafKind: LF_PROCEDURE (0x1008)
; CHECK:     ReturnType: BUnion ({{.*}})
; CHECK:     CallingConvention: NearC (0x0)
; CHECK:     FunctionOptions [ (0x0)
; CHECK:     ]
; CHECK:     NumParameters: 1
; CHECK:     ArgListType: (BUnion&) ({{.*}})
; CHECK:   }
; CHECK:   MemberFunction ([[CTOR_BU:.*]]) {
; CHECK:     TypeLeafKind: LF_MFUNCTION (0x1009)
; CHECK:     ReturnType: void (0x3)
; CHECK:     ClassType: BUnion ({{.*}})
; CHECK:     ThisType: BUnion* {{.*}}
; CHECK:     CallingConvention: NearC (0x0)
; CHECK:     FunctionOptions [ (0x0)
; CHECK:     ]
; CHECK:     NumParameters: 0
; CHECK:     ArgListType: () ({{.*}})
; CHECK:     ThisAdjustment: 0
; CHECK:   }
; CHECK:   FieldList ({{.*}}) {
; CHECK:     TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK:     OneMethod {
; CHECK:       TypeLeafKind: LF_ONEMETHOD (0x1511)
; CHECK:       AccessSpecifier: Public (0x3)
; CHECK:       Type: void BUnion::() ([[CTOR_BU]])
; CHECK:       Name: BUnion
; CHECK:     }
; CHECK:   }
; CHECK:   FuncId ({{.*}}) {
; CHECK:     TypeLeafKind: LF_FUNC_ID (0x1601)
; CHECK:     ParentScope: 0x0
; CHECK:     FunctionType: BUnion (BUnion&) ([[SP_BU]])
; CHECK:     Name: Func_BUnion
; CHECK:   }
; CHECK: ]

; CHECK-LABEL: FunctionType: Func_AClass (0x1008)
; CHECK-NEXT:  CodeOffset: ?Func_AClass@@YA?AVAClass@@AEAV1@@Z+0x0
; CHECK-NEXT:  Segment: 0x0
; CHECK-NEXT:  Flags [ (0xC0)
; CHECK-NEXT:    HasOptimizedDebugInfo (0x80)
; CHECK-NEXT:    IsNoInline (0x40)
; CHECK-NEXT:  ]
; CHECK-NEXT:  DisplayName: Func_AClass
; CHECK-NEXT:  LinkageName: ?Func_AClass@@YA?AVAClass@@AEAV1@@Z
; CHECK-LABEL: FunctionType: Func_BClass (0x1013)
; CHECK-NEXT:  CodeOffset: ?Func_BClass@@YA?AVBClass@@AEAV1@@Z+0x0
; CHECK-NEXT:  Segment: 0x0
; CHECK-NEXT:  Flags [ (0xC0)
; CHECK-NEXT:    HasOptimizedDebugInfo (0x80)
; CHECK-NEXT:    IsNoInline (0x40)
; CHECK-NEXT:  ]
; CHECK-NEXT:  DisplayName: Func_BClass
; CHECK-NEXT:  LinkageName: ?Func_BClass@@YA?AVBClass@@AEAV1@@Z
; CHECK-LABEL: FunctionType: Func_C1Class (0x101D)
; CHECK-NEXT:  CodeOffset: ?Func_C1Class@@YA?AVC1Class@@AEAV1@@Z+0x0
; CHECK-NEXT:  Segment: 0x0
; CHECK-NEXT:  Flags [ (0xC0)
; CHECK-NEXT:    HasOptimizedDebugInfo (0x80)
; CHECK-NEXT:    IsNoInline (0x40)
; CHECK-NEXT:  ]
; CHECK-NEXT:  DisplayName: Func_C1Class
; CHECK-NEXT:  LinkageName: ?Func_C1Class@@YA?AVC1Class@@AEAV1@@Z
; CHECK-LABEL: FunctionType: Func_C2Class (0x1027)
; CHECK-NEXT:  CodeOffset: ?Func_C2Class@@YA?AVC2Class@@AEAV1@@Z+0x0
; CHECK-NEXT:  Segment: 0x0
; CHECK-NEXT:  Flags [ (0xC0)
; CHECK-NEXT:    HasOptimizedDebugInfo (0x80)
; CHECK-NEXT:    IsNoInline (0x40)
; CHECK-NEXT:  ]
; CHECK-NEXT:  DisplayName: Func_C2Class
; CHECK-NEXT:  LinkageName: ?Func_C2Class@@YA?AVC2Class@@AEAV1@@Z
; CHECK-LABEL: FunctionType: Func_DClass (0x102F)
; CHECK-NEXT:  CodeOffset: ?Func_DClass@@YA?AVDClass@@AEAV1@@Z+0x0
; CHECK-NEXT:  Segment: 0x0
; CHECK-NEXT:  Flags [ (0xC0)
; CHECK-NEXT:    HasOptimizedDebugInfo (0x80)
; CHECK-NEXT:    IsNoInline (0x40)
; CHECK-NEXT:  ]
; CHECK-NEXT:  DisplayName: Func_DClass
; CHECK-NEXT:  LinkageName: ?Func_DClass@@YA?AVDClass@@AEAV1@@Z
; CHECK-LABEL: FunctionType: Func_FClass (0x103A)
; CHECK-NEXT:  CodeOffset: ?Func_FClass@@YA?AVFClass@@AEAV1@@Z+0x0
; CHECK-NEXT:  Segment: 0x0
; CHECK-NEXT:  Flags [ (0xC0)
; CHECK-NEXT:    HasOptimizedDebugInfo (0x80)
; CHECK-NEXT:    IsNoInline (0x40)
; CHECK-NEXT:  ]
; CHECK-NEXT:  DisplayName: Func_FClass
; CHECK-NEXT:  LinkageName: ?Func_FClass@@YA?AVFClass@@AEAV1@@Z
; CHECK-LABEL: FunctionType: Func_AStruct (0x1041)
; CHECK-NEXT:  CodeOffset: ?Func_AStruct@@YA?AUAStruct@@AEAU1@@Z+0x0
; CHECK-NEXT:  Segment: 0x0
; CHECK-NEXT:  Flags [ (0xC0)
; CHECK-NEXT:    HasOptimizedDebugInfo (0x80)
; CHECK-NEXT:    IsNoInline (0x40)
; CHECK-NEXT:  ]
; CHECK-NEXT:  DisplayName: Func_AStruct
; CHECK-NEXT:  LinkageName: ?Func_AStruct@@YA?AUAStruct@@AEAU1@@Z
; CHECK-LABEL: FunctionType: Func_BStruct (0x104B)
; CHECK-NEXT:  CodeOffset: ?Func_BStruct@@YA?AUBStruct@@AEAU1@@Z+0x0
; CHECK-NEXT:  Segment: 0x0
; CHECK-NEXT:  Flags [ (0xC0)
; CHECK-NEXT:    HasOptimizedDebugInfo (0x80)
; CHECK-NEXT:    IsNoInline (0x40)
; CHECK-NEXT:  ]
; CHECK-NEXT:  DisplayName: Func_BStruct
; CHECK-NEXT:  LinkageName: ?Func_BStruct@@YA?AUBStruct@@AEAU1@@Z
; CHECK-LABEL: FunctionType: Func_AUnion (0x1052)
; CHECK-NEXT:  CodeOffset: ?Func_AUnion@@YA?ATAUnion@@AEAT1@@Z+0x0
; CHECK-NEXT:  Segment: 0x0
; CHECK-NEXT:  Flags [ (0xC0)
; CHECK-NEXT:    HasOptimizedDebugInfo (0x80)
; CHECK-NEXT:    IsNoInline (0x40)
; CHECK-NEXT:  ]
; CHECK-NEXT:  DisplayName: Func_AUnion
; CHECK-NEXT:  LinkageName: ?Func_AUnion@@YA?ATAUnion@@AEAT1@@Z
; CHECK-LABEL: FunctionType: Func_BUnion (0x105C)
; CHECK-NEXT:  CodeOffset: ?Func_BUnion@@YA?ATBUnion@@AEAT1@@Z+0x0
; CHECK-NEXT:  Segment: 0x0
; CHECK-NEXT:  Flags [ (0xC0)
; CHECK-NEXT:    HasOptimizedDebugInfo (0x80)
; CHECK-NEXT:    IsNoInline (0x40)
; CHECK-NEXT:  ]
; CHECK-NEXT:  DisplayName: Func_BUnion
; CHECK-NEXT:  LinkageName: ?Func_BUnion@@YA?ATBUnion@@AEAT1@@Z

; ModuleID = 't.cpp'
source_filename = "t.cpp"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.23.28106"

%class.AClass = type { i8 }
%class.BClass = type { i8 }
%class.C1Class = type { i8 }
%class.C2Class = type { i8 }
%class.DClass = type { i8 }
%class.FClass = type { i8 }
%struct.AStruct = type { i8 }
%struct.BStruct = type { i8 }
%union.AUnion = type { i8 }
%union.BUnion = type { i8 }

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i8 @"?Func_AClass@@YA?AVAClass@@AEAV1@@Z"(%class.AClass* dereferenceable(1) %arg) #0 !dbg !8 {
entry:
  %retval = alloca %class.AClass, align 1
  %arg.addr = alloca %class.AClass*, align 8
  store %class.AClass* %arg, %class.AClass** %arg.addr, align 8
  call void @llvm.dbg.declare(metadata %class.AClass** %arg.addr, metadata !13, metadata !DIExpression()), !dbg !14
  %0 = load %class.AClass*, %class.AClass** %arg.addr, align 8, !dbg !14
  %coerce.dive = getelementptr inbounds %class.AClass, %class.AClass* %retval, i32 0, i32 0, !dbg !14
  %1 = load i8, i8* %coerce.dive, align 1, !dbg !14
  ret i8 %1, !dbg !14
}

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @"?Func_BClass@@YA?AVBClass@@AEAV1@@Z"(%class.BClass* noalias sret(%class.BClass) %agg.result, %class.BClass* dereferenceable(1) %arg) #0 !dbg !15 {
entry:
  %result.ptr = alloca i8*, align 8
  %arg.addr = alloca %class.BClass*, align 8
  %0 = bitcast %class.BClass* %agg.result to i8*
  store i8* %0, i8** %result.ptr, align 8
  store %class.BClass* %arg, %class.BClass** %arg.addr, align 8
  call void @llvm.dbg.declare(metadata %class.BClass** %arg.addr, metadata !25, metadata !DIExpression()), !dbg !26
  %1 = load %class.BClass*, %class.BClass** %arg.addr, align 8, !dbg !26
  ret void, !dbg !26
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @"?Func_C1Class@@YA?AVC1Class@@AEAV1@@Z"(%class.C1Class* noalias sret(%class.C1Class) %agg.result, %class.C1Class* dereferenceable(1) %arg) #0 !dbg !27 {
entry:
  %result.ptr = alloca i8*, align 8
  %arg.addr = alloca %class.C1Class*, align 8
  %0 = bitcast %class.C1Class* %agg.result to i8*
  store i8* %0, i8** %result.ptr, align 8
  store %class.C1Class* %arg, %class.C1Class** %arg.addr, align 8
  call void @llvm.dbg.declare(metadata %class.C1Class** %arg.addr, metadata !37, metadata !DIExpression()), !dbg !38
  %1 = load %class.C1Class*, %class.C1Class** %arg.addr, align 8, !dbg !38
  ret void, !dbg !38
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @"?Func_C2Class@@YA?AVC2Class@@AEAV1@@Z"(%class.C2Class* noalias sret(%class.C2Class) %agg.result, %class.C2Class* dereferenceable(1) %arg) #0 !dbg !39 {
entry:
  %result.ptr = alloca i8*, align 8
  %arg.addr = alloca %class.C2Class*, align 8
  %0 = bitcast %class.C2Class* %agg.result to i8*
  store i8* %0, i8** %result.ptr, align 8
  store %class.C2Class* %arg, %class.C2Class** %arg.addr, align 8
  call void @llvm.dbg.declare(metadata %class.C2Class** %arg.addr, metadata !49, metadata !DIExpression()), !dbg !50
  %1 = load %class.C2Class*, %class.C2Class** %arg.addr, align 8, !dbg !50
  ret void, !dbg !50
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @"?Func_DClass@@YA?AVDClass@@AEAV1@@Z"(%class.DClass* noalias sret(%class.DClass) %agg.result, %class.DClass* dereferenceable(1) %arg) #0 !dbg !51 {
entry:
  %result.ptr = alloca i8*, align 8
  %arg.addr = alloca %class.DClass*, align 8
  %0 = bitcast %class.DClass* %agg.result to i8*
  store i8* %0, i8** %result.ptr, align 8
  store %class.DClass* %arg, %class.DClass** %arg.addr, align 8
  call void @llvm.dbg.declare(metadata %class.DClass** %arg.addr, metadata !58, metadata !DIExpression()), !dbg !59
  %1 = load %class.DClass*, %class.DClass** %arg.addr, align 8, !dbg !59
  ret void, !dbg !59
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i8 @"?Func_FClass@@YA?AVFClass@@AEAV1@@Z"(%class.FClass* dereferenceable(1) %arg) #0 !dbg !60 {
entry:
  %retval = alloca %class.FClass, align 1
  %arg.addr = alloca %class.FClass*, align 8
  store %class.FClass* %arg, %class.FClass** %arg.addr, align 8
  call void @llvm.dbg.declare(metadata %class.FClass** %arg.addr, metadata !75, metadata !DIExpression()), !dbg !76
  %0 = load %class.FClass*, %class.FClass** %arg.addr, align 8, !dbg !76
  %coerce.dive = getelementptr inbounds %class.FClass, %class.FClass* %retval, i32 0, i32 0, !dbg !76
  %1 = load i8, i8* %coerce.dive, align 1, !dbg !76
  ret i8 %1, !dbg !76
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i8 @"?Func_AStruct@@YA?AUAStruct@@AEAU1@@Z"(%struct.AStruct* dereferenceable(1) %arg) #0 !dbg !77 {
entry:
  %retval = alloca %struct.AStruct, align 1
  %arg.addr = alloca %struct.AStruct*, align 8
  store %struct.AStruct* %arg, %struct.AStruct** %arg.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.AStruct** %arg.addr, metadata !82, metadata !DIExpression()), !dbg !83
  %0 = load %struct.AStruct*, %struct.AStruct** %arg.addr, align 8, !dbg !83
  %coerce.dive = getelementptr inbounds %struct.AStruct, %struct.AStruct* %retval, i32 0, i32 0, !dbg !83
  %1 = load i8, i8* %coerce.dive, align 1, !dbg !83
  ret i8 %1, !dbg !83
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @"?Func_BStruct@@YA?AUBStruct@@AEAU1@@Z"(%struct.BStruct* noalias sret(%struct.BStruct) %agg.result, %struct.BStruct* dereferenceable(1) %arg) #0 !dbg !84 {
entry:
  %result.ptr = alloca i8*, align 8
  %arg.addr = alloca %struct.BStruct*, align 8
  %0 = bitcast %struct.BStruct* %agg.result to i8*
  store i8* %0, i8** %result.ptr, align 8
  store %struct.BStruct* %arg, %struct.BStruct** %arg.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.BStruct** %arg.addr, metadata !94, metadata !DIExpression()), !dbg !95
  %1 = load %struct.BStruct*, %struct.BStruct** %arg.addr, align 8, !dbg !95
  ret void, !dbg !95
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i8 @"?Func_AUnion@@YA?ATAUnion@@AEAT1@@Z"(%union.AUnion* dereferenceable(1) %arg) #0 !dbg !96 {
entry:
  %retval = alloca %union.AUnion, align 1
  %arg.addr = alloca %union.AUnion*, align 8
  store %union.AUnion* %arg, %union.AUnion** %arg.addr, align 8
  call void @llvm.dbg.declare(metadata %union.AUnion** %arg.addr, metadata !101, metadata !DIExpression()), !dbg !102
  %0 = load %union.AUnion*, %union.AUnion** %arg.addr, align 8, !dbg !102
  %coerce.dive = getelementptr inbounds %union.AUnion, %union.AUnion* %retval, i32 0, i32 0, !dbg !102
  %1 = load i8, i8* %coerce.dive, align 1, !dbg !102
  ret i8 %1, !dbg !102
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @"?Func_BUnion@@YA?ATBUnion@@AEAT1@@Z"(%union.BUnion* noalias sret(%union.BUnion) %agg.result, %union.BUnion* dereferenceable(1) %arg) #0 !dbg !103 {
entry:
  %result.ptr = alloca i8*, align 8
  %arg.addr = alloca %union.BUnion*, align 8
  %0 = bitcast %union.BUnion* %agg.result to i8*
  store i8* %0, i8** %result.ptr, align 8
  store %union.BUnion* %arg, %union.BUnion** %arg.addr, align 8
  call void @llvm.dbg.declare(metadata %union.BUnion** %arg.addr, metadata !113, metadata !DIExpression()), !dbg !114
  %1 = load %union.BUnion*, %union.BUnion** %arg.addr, align 8, !dbg !114
  ret void, !dbg !114
}

; Function Attrs: noinline norecurse nounwind optnone uwtable
define dso_local i32 @main() #2 !dbg !115 {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  ret i32 0, !dbg !118
}

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable willreturn }
attributes #2 = { noinline norecurse nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 11.0.0 (https://github.com/llvm/llvm-project.git 48992717b0e3466cf8814a188e9568c9d71b59c2)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "t.cpp", directory: "C:\\src\\tests\\duplicate-types\\llvm-test", checksumkind: CSK_MD5, checksum: "c4c61c0e2135d713d0c99a1ba9ab568b")
!2 = !{}
!3 = !{i32 2, !"CodeView", i32 1}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 2}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = !{!"clang version 11.0.0 (https://github.com/llvm/llvm-project.git 48992717b0e3466cf8814a188e9568c9d71b59c2)"}
!8 = distinct !DISubprogram(name: "Func_AClass", linkageName: "?Func_AClass@@YA?AVAClass@@AEAV1@@Z", scope: !1, file: !1, line: 5, type: !9, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!9 = !DISubroutineType(types: !10)
!10 = !{!11, !12}
!11 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "AClass", file: !1, line: 4, size: 8, flags: DIFlagTypePassByValue, elements: !2, identifier: ".?AVAClass@@")
!12 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !11, size: 64)
!13 = !DILocalVariable(name: "arg", arg: 1, scope: !8, file: !1, line: 5, type: !12)
!14 = !DILocation(line: 5, scope: !8)
!15 = distinct !DISubprogram(name: "Func_BClass", linkageName: "?Func_BClass@@YA?AVBClass@@AEAV1@@Z", scope: !1, file: !1, line: 11, type: !16, scopeLine: 11, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!16 = !DISubroutineType(types: !17)
!17 = !{!18, !24}
!18 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "BClass", file: !1, line: 7, size: 8, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !19, identifier: ".?AVBClass@@")
!19 = !{!20}
!20 = !DISubprogram(name: "BClass", scope: !18, file: !1, line: 9, type: !21, scopeLine: 9, flags: DIFlagExplicit | DIFlagPrototyped, spFlags: 0)
!21 = !DISubroutineType(types: !22)
!22 = !{null, !23}
!23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !18, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!24 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !18, size: 64)
!25 = !DILocalVariable(name: "arg", arg: 1, scope: !15, file: !1, line: 11, type: !24)
!26 = !DILocation(line: 11, scope: !15)
!27 = distinct !DISubprogram(name: "Func_C1Class", linkageName: "?Func_C1Class@@YA?AVC1Class@@AEAV1@@Z", scope: !1, file: !1, line: 17, type: !28, scopeLine: 17, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!28 = !DISubroutineType(types: !29)
!29 = !{!30, !36}
!30 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "C1Class", file: !1, line: 13, size: 8, flags: DIFlagTypePassByValue, elements: !31, identifier: ".?AVC1Class@@")
!31 = !{!32}
!32 = !DISubprogram(name: "C1Class", scope: !30, file: !1, line: 15, type: !33, scopeLine: 15, flags: DIFlagPublic | DIFlagPrototyped, spFlags: 0)
!33 = !DISubroutineType(types: !34)
!34 = !{null, !35}
!35 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !30, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!36 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !30, size: 64)
!37 = !DILocalVariable(name: "arg", arg: 1, scope: !27, file: !1, line: 17, type: !36)
!38 = !DILocation(line: 17, scope: !27)
!39 = distinct !DISubprogram(name: "Func_C2Class", linkageName: "?Func_C2Class@@YA?AVC2Class@@AEAV1@@Z", scope: !1, file: !1, line: 23, type: !40, scopeLine: 23, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!40 = !DISubroutineType(types: !41)
!41 = !{!42, !48}
!42 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "C2Class", file: !1, line: 19, size: 8, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !43, identifier: ".?AVC2Class@@")
!43 = !{!44}
!44 = !DISubprogram(name: "~C2Class", scope: !42, file: !1, line: 21, type: !45, scopeLine: 21, flags: DIFlagPublic | DIFlagPrototyped, spFlags: 0)
!45 = !DISubroutineType(types: !46)
!46 = !{null, !47}
!47 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !42, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!48 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !42, size: 64)
!49 = !DILocalVariable(name: "arg", arg: 1, scope: !39, file: !1, line: 23, type: !48)
!50 = !DILocation(line: 23, scope: !39)
!51 = distinct !DISubprogram(name: "Func_DClass", linkageName: "?Func_DClass@@YA?AVDClass@@AEAV1@@Z", scope: !1, file: !1, line: 26, type: !52, scopeLine: 26, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!52 = !DISubroutineType(types: !53)
!53 = !{!54, !57}
!54 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "DClass", file: !1, line: 25, size: 8, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !55, identifier: ".?AVDClass@@")
!55 = !{!56}
!56 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !54, baseType: !18, flags: DIFlagPublic, extraData: i32 0)
!57 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !54, size: 64)
!58 = !DILocalVariable(name: "arg", arg: 1, scope: !51, file: !1, line: 26, type: !57)
!59 = !DILocation(line: 26, scope: !51)
!60 = distinct !DISubprogram(name: "Func_FClass", linkageName: "?Func_FClass@@YA?AVFClass@@AEAV1@@Z", scope: !1, file: !1, line: 33, type: !61, scopeLine: 33, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!61 = !DISubroutineType(types: !62)
!62 = !{!63, !74}
!63 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "FClass", file: !1, line: 28, size: 8, flags: DIFlagTypePassByValue, elements: !64, identifier: ".?AVFClass@@")
!64 = !{!65, !67, !71}
!65 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !63, file: !1, line: 29, baseType: !66, flags: DIFlagStaticMember)
!66 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!67 = !DISubprogram(name: "Member_AClass", linkageName: "?Member_AClass@FClass@@AEAA?AVAClass@@AEAV2@@Z", scope: !63, file: !1, line: 30, type: !68, scopeLine: 30, flags: DIFlagPrototyped, spFlags: 0)
!68 = !DISubroutineType(types: !69)
!69 = !{!11, !70, !12}
!70 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !63, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!71 = !DISubprogram(name: "Member_BClass", linkageName: "?Member_BClass@FClass@@AEAA?AVBClass@@AEAV2@@Z", scope: !63, file: !1, line: 31, type: !72, scopeLine: 31, flags: DIFlagPrototyped, spFlags: 0)
!72 = !DISubroutineType(types: !73)
!73 = !{!18, !70, !24}
!74 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !63, size: 64)
!75 = !DILocalVariable(name: "arg", arg: 1, scope: !60, file: !1, line: 33, type: !74)
!76 = !DILocation(line: 33, scope: !60)
!77 = distinct !DISubprogram(name: "Func_AStruct", linkageName: "?Func_AStruct@@YA?AUAStruct@@AEAU1@@Z", scope: !1, file: !1, line: 36, type: !78, scopeLine: 36, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!78 = !DISubroutineType(types: !79)
!79 = !{!80, !81}
!80 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "AStruct", file: !1, line: 35, size: 8, flags: DIFlagTypePassByValue, elements: !2, identifier: ".?AUAStruct@@")
!81 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !80, size: 64)
!82 = !DILocalVariable(name: "arg", arg: 1, scope: !77, file: !1, line: 36, type: !81)
!83 = !DILocation(line: 36, scope: !77)
!84 = distinct !DISubprogram(name: "Func_BStruct", linkageName: "?Func_BStruct@@YA?AUBStruct@@AEAU1@@Z", scope: !1, file: !1, line: 39, type: !85, scopeLine: 39, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!85 = !DISubroutineType(types: !86)
!86 = !{!87, !93}
!87 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "BStruct", file: !1, line: 38, size: 8, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !88, identifier: ".?AUBStruct@@")
!88 = !{!89}
!89 = !DISubprogram(name: "BStruct", scope: !87, file: !1, line: 38, type: !90, scopeLine: 38, flags: DIFlagPrototyped, spFlags: 0)
!90 = !DISubroutineType(types: !91)
!91 = !{null, !92}
!92 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !87, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!93 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !87, size: 64)
!94 = !DILocalVariable(name: "arg", arg: 1, scope: !84, file: !1, line: 39, type: !93)
!95 = !DILocation(line: 39, scope: !84)
!96 = distinct !DISubprogram(name: "Func_AUnion", linkageName: "?Func_AUnion@@YA?ATAUnion@@AEAT1@@Z", scope: !1, file: !1, line: 42, type: !97, scopeLine: 42, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!97 = !DISubroutineType(types: !98)
!98 = !{!99, !100}
!99 = distinct !DICompositeType(tag: DW_TAG_union_type, name: "AUnion", file: !1, line: 41, size: 8, flags: DIFlagTypePassByValue, elements: !2, identifier: ".?ATAUnion@@")
!100 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !99, size: 64)
!101 = !DILocalVariable(name: "arg", arg: 1, scope: !96, file: !1, line: 42, type: !100)
!102 = !DILocation(line: 42, scope: !96)
!103 = distinct !DISubprogram(name: "Func_BUnion", linkageName: "?Func_BUnion@@YA?ATBUnion@@AEAT1@@Z", scope: !1, file: !1, line: 45, type: !104, scopeLine: 45, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!104 = !DISubroutineType(types: !105)
!105 = !{!106, !112}
!106 = distinct !DICompositeType(tag: DW_TAG_union_type, name: "BUnion", file: !1, line: 44, size: 8, flags: DIFlagTypePassByValue, elements: !107, identifier: ".?ATBUnion@@")
!107 = !{!108}
!108 = !DISubprogram(name: "BUnion", scope: !106, file: !1, line: 44, type: !109, scopeLine: 44, flags: DIFlagPrototyped, spFlags: 0)
!109 = !DISubroutineType(types: !110)
!110 = !{null, !111}
!111 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !106, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!112 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !106, size: 64)
!113 = !DILocalVariable(name: "arg", arg: 1, scope: !103, file: !1, line: 45, type: !112)
!114 = !DILocation(line: 45, scope: !103)
!115 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 47, type: !116, scopeLine: 47, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!116 = !DISubroutineType(types: !117)
!117 = !{!66}
!118 = !DILocation(line: 48, scope: !115)
