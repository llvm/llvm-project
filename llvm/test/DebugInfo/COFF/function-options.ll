; RUN: llc < %s -filetype=obj | llvm-readobj - -codeview | FileCheck %s
;
; Command to generate function-options.ll
; $ clang++ function-options.cpp -S -emit-llvm -g -gcodeview -o function-options.ll
;
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
; class FClass { static int x; };
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
; CHECK:   Procedure ([[SP1:.*]]) {
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
; CHECK:     FunctionType: AClass (AClass&) ([[SP1]])
; CHECK:     Name: Func_AClass
; CHECK:   }
; CHECK:   Procedure ([[SP2:.*]]) {
; CHECK:     TypeLeafKind: LF_PROCEDURE (0x1008)
; CHECK:     ReturnType: BClass ({{.*}})
; CHECK:     CallingConvention: NearC (0x0)
; CHECK:     FunctionOptions [ (0x1)
; CHECK:       CxxReturnUdt (0x1)
; CHECK:     ]
; CHECK:     NumParameters: 1
; CHECK:     ArgListType: (BClass&) ({{.*}})
; CHECK:   }
; CHECK:   MemberFunction ([[MF1:.*]]) {
; CHECK:     TypeLeafKind: LF_MFUNCTION (0x1009)
; CHECK:     ReturnType: void (0x3)
; CHECK:     ClassType: BClass ({{.*}})
; CHECK:     ThisType: BClass* const ({{.*}})
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
; CHECK:       Type: void BClass::() ([[MF1]])
; CHECK:       Name: BClass
; CHECK:     }
; CHECK:   }
; CHECK:   FuncId ({{.*}}) {
; CHECK:     TypeLeafKind: LF_FUNC_ID (0x1601)
; CHECK:     ParentScope: 0x0
; CHECK:     FunctionType: BClass (BClass&) ([[SP2]])
; CHECK:     Name: Func_BClass
; CHECK:   }
; CHECK:   Procedure ([[SP3:.*]]) {
; CHECK:     TypeLeafKind: LF_PROCEDURE (0x1008)
; CHECK:     ReturnType: C1Class ({{.*}})
; CHECK:     CallingConvention: NearC (0x0)
; CHECK:     FunctionOptions [ (0x0)
; CHECK:     ]
; CHECK:     NumParameters: 1
; CHECK:     ArgListType: (C1Class&) ({{.*}})
; CHECK:   }
; CHECK:   MemberFunction ([[MF2:.*]]) {
; CHECK:     TypeLeafKind: LF_MFUNCTION (0x1009)
; CHECK:     ReturnType: void (0x3)
; CHECK:     ClassType: C1Class ({{.*}})
; CHECK:     ThisType: C1Class* const ({{.*}})
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
; CHECK:       Type: void C1Class::() ([[MF2]])
; CHECK:       Name: C1Class
; CHECK:     }
; CHECK:   }
; CHECK:   FuncId ({{.*}}) {
; CHECK:     TypeLeafKind: LF_FUNC_ID (0x1601)
; CHECK:     ParentScope: 0x0
; CHECK:     FunctionType: C1Class (C1Class&) ([[SP3]])
; CHECK:     Name: Func_C1Class
; CHECK:   }
; CHECK:   Procedure ([[SP4:.*]]) {
; CHECK:     TypeLeafKind: LF_PROCEDURE (0x1008)
; CHECK:     ReturnType: C2Class ({{.*}})
; CHECK:     CallingConvention: NearC (0x0)
; CHECK:     FunctionOptions [ (0x1)
; CHECK:       CxxReturnUdt (0x1)
; CHECK:     ]
; CHECK:     NumParameters: 1
; CHECK:     ArgListType: (C2Class&) ({{.*}})
; CHECK:   }
; CHECK:   MemberFunction ([[MF3:.*]]) {
; CHECK:     TypeLeafKind: LF_MFUNCTION (0x1009)
; CHECK:     ReturnType: void (0x3)
; CHECK:     ClassType: C2Class ({{.*}})
; CHECK:     ThisType: C2Class* const ({{.*}})
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
; CHECK:       Type: void C2Class::() ([[MF3]])
; CHECK:       Name: ~C2Class
; CHECK:     }
; CHECK:   }
; CHECK:   FuncId ({{.*}}) {
; CHECK:     TypeLeafKind: LF_FUNC_ID (0x1601)
; CHECK:     ParentScope: 0x0
; CHECK:     FunctionType: C2Class (C2Class&) ([[SP4]])
; CHECK:     Name: Func_C2Class
; CHECK:   }
; CHECK:   Procedure ([[SP5:.*]]) {
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
; CHECK:     FunctionType: DClass (DClass&) ([[SP5]])
; CHECK:     Name: Func_DClass
; CHECK:   }
; CHECK:   Procedure ([[SP6:.*]]) {
; CHECK:     TypeLeafKind: LF_PROCEDURE (0x1008)
; CHECK:     ReturnType: FClass ({{.*}})
; CHECK:     CallingConvention: NearC (0x0)
; CHECK:     FunctionOptions [ (0x0)
; CHECK:     ]
; CHECK:     NumParameters: 1
; CHECK:     ArgListType: (FClass&) ({{.*}})
; CHECK:   }
; CHECK:   FieldList ({{.*}}) {
; CHECK:     TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK:     StaticDataMember {
; CHECK:       TypeLeafKind: LF_STMEMBER (0x150E)
; CHECK:       AccessSpecifier: Private (0x1)
; CHECK:       Type: int (0x74)
; CHECK:       Name: x
; CHECK:     }
; CHECK:   }
; CHECK:   FuncId ({{.*}}) {
; CHECK:     TypeLeafKind: LF_FUNC_ID (0x1601)
; CHECK:     ParentScope: 0x0
; CHECK:     FunctionType: FClass (FClass&) ([[SP6]])
; CHECK:     Name: Func_FClass
; CHECK:   }
; CHECK:   Procedure ([[SP7:.*]]) {
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
; CHECK:     FunctionType: AStruct (AStruct&) ([[SP7]])
; CHECK:     Name: Func_AStruct
; CHECK:   }
; CHECK:   Procedure ([[SP8:.*]]) {
; CHECK:     TypeLeafKind: LF_PROCEDURE (0x1008)
; CHECK:     ReturnType: BStruct ({{.*}})
; CHECK:     CallingConvention: NearC (0x0)
; CHECK:     FunctionOptions [ (0x1)
; CHECK:       CxxReturnUdt (0x1)
; CHECK:     ]
; CHECK:     NumParameters: 1
; CHECK:     ArgListType: (BStruct&) ({{.*}})
; CHECK:   }
; CHECK:   MemberFunction ([[MF4:.*]]) {
; CHECK:     TypeLeafKind: LF_MFUNCTION (0x1009)
; CHECK:     ReturnType: void (0x3)
; CHECK:     ClassType: BStruct ({{.*}})
; CHECK:     ThisType: BStruct* const ({{.*}})
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
; CHECK:       Type: void BStruct::() ([[MF4]])
; CHECK:       Name: BStruct
; CHECK:     }
; CHECK:   }
; CHECK:   FuncId ({{.*}}) {
; CHECK:     TypeLeafKind: LF_FUNC_ID (0x1601)
; CHECK:     ParentScope: 0x0
; CHECK:     FunctionType: BStruct (BStruct&) ([[SP8]])
; CHECK:     Name: Func_BStruct
; CHECK:   }
; CHECK:   Procedure ([[SP9:.*]]) {
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
; CHECK:     FunctionType: AUnion (AUnion&) ([[SP9]])
; CHECK:     Name: Func_AUnion
; CHECK:   }
; CHECK:   Procedure ([[SP10:.*]]) {
; CHECK:     TypeLeafKind: LF_PROCEDURE (0x1008)
; CHECK:     ReturnType: BUnion ({{.*}})
; CHECK:     CallingConvention: NearC (0x0)
; CHECK:     FunctionOptions [ (0x0)
; CHECK:     ]
; CHECK:     NumParameters: 1
; CHECK:     ArgListType: (BUnion&) ({{.*}})
; CHECK:   }
; CHECK:   MemberFunction ([[MF5:.*]]) {
; CHECK:     TypeLeafKind: LF_MFUNCTION (0x1009)
; CHECK:     ReturnType: void (0x3)
; CHECK:     ClassType: BUnion ({{.*}})
; CHECK:     ThisType: BUnion* const ({{.*}})
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
; CHECK:       Type: void BUnion::() ([[MF5]])
; CHECK:       Name: BUnion
; CHECK:     }
; CHECK:   }
; CHECK:   FuncId ({{.*}}) {
; CHECK:     TypeLeafKind: LF_FUNC_ID (0x1601)
; CHECK:     ParentScope: 0x0
; CHECK:     FunctionType: BUnion (BUnion&) ([[SP10]])
; CHECK:     Name: Func_BUnion
; CHECK:   }
; CHECK: ]


; ModuleID = 'function-options.cpp'
source_filename = "function-options.cpp"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.15.26729"

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
  call void @llvm.dbg.declare(metadata %class.AClass** %arg.addr, metadata !14, metadata !DIExpression()), !dbg !15
  %0 = load %class.AClass*, %class.AClass** %arg.addr, align 8, !dbg !15
  %coerce.dive = getelementptr inbounds %class.AClass, %class.AClass* %retval, i32 0, i32 0, !dbg !15
  %1 = load i8, i8* %coerce.dive, align 1, !dbg !15
  ret i8 %1, !dbg !15
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @"?Func_BClass@@YA?AVBClass@@AEAV1@@Z"(%class.BClass* noalias sret %agg.result, %class.BClass* dereferenceable(1) %arg) #0 !dbg !16 {
entry:
  %arg.addr = alloca %class.BClass*, align 8
  store %class.BClass* %arg, %class.BClass** %arg.addr, align 8
  call void @llvm.dbg.declare(metadata %class.BClass** %arg.addr, metadata !26, metadata !DIExpression()), !dbg !27
  %0 = load %class.BClass*, %class.BClass** %arg.addr, align 8, !dbg !27
  ret void, !dbg !27
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @"?Func_C1Class@@YA?AVC1Class@@AEAV1@@Z"(%class.C1Class* noalias sret %agg.result, %class.C1Class* dereferenceable(1) %arg) #0 !dbg !28 {
entry:
  %arg.addr = alloca %class.C1Class*, align 8
  store %class.C1Class* %arg, %class.C1Class** %arg.addr, align 8
  call void @llvm.dbg.declare(metadata %class.C1Class** %arg.addr, metadata !38, metadata !DIExpression()), !dbg !39
  %0 = load %class.C1Class*, %class.C1Class** %arg.addr, align 8, !dbg !39
  ret void, !dbg !39
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @"?Func_C2Class@@YA?AVC2Class@@AEAV1@@Z"(%class.C2Class* noalias sret %agg.result, %class.C2Class* dereferenceable(1) %arg) #0 !dbg !40 {
entry:
  %arg.addr = alloca %class.C2Class*, align 8
  store %class.C2Class* %arg, %class.C2Class** %arg.addr, align 8
  call void @llvm.dbg.declare(metadata %class.C2Class** %arg.addr, metadata !50, metadata !DIExpression()), !dbg !51
  %0 = load %class.C2Class*, %class.C2Class** %arg.addr, align 8, !dbg !51
  ret void, !dbg !51
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @"?Func_DClass@@YA?AVDClass@@AEAV1@@Z"(%class.DClass* noalias sret %agg.result, %class.DClass* dereferenceable(1) %arg) #0 !dbg !52 {
entry:
  %arg.addr = alloca %class.DClass*, align 8
  store %class.DClass* %arg, %class.DClass** %arg.addr, align 8
  call void @llvm.dbg.declare(metadata %class.DClass** %arg.addr, metadata !59, metadata !DIExpression()), !dbg !60
  %0 = load %class.DClass*, %class.DClass** %arg.addr, align 8, !dbg !60
  ret void, !dbg !60
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i8 @"?Func_FClass@@YA?AVFClass@@AEAV1@@Z"(%class.FClass* dereferenceable(1) %arg) #0 !dbg !61 {
entry:
  %retval = alloca %class.FClass, align 1
  %arg.addr = alloca %class.FClass*, align 8
  store %class.FClass* %arg, %class.FClass** %arg.addr, align 8
  call void @llvm.dbg.declare(metadata %class.FClass** %arg.addr, metadata !69, metadata !DIExpression()), !dbg !70
  %0 = load %class.FClass*, %class.FClass** %arg.addr, align 8, !dbg !70
  %coerce.dive = getelementptr inbounds %class.FClass, %class.FClass* %retval, i32 0, i32 0, !dbg !70
  %1 = load i8, i8* %coerce.dive, align 1, !dbg !70
  ret i8 %1, !dbg !70
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i8 @"?Func_AStruct@@YA?AUAStruct@@AEAU1@@Z"(%struct.AStruct* dereferenceable(1) %arg) #0 !dbg !71 {
entry:
  %retval = alloca %struct.AStruct, align 1
  %arg.addr = alloca %struct.AStruct*, align 8
  store %struct.AStruct* %arg, %struct.AStruct** %arg.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.AStruct** %arg.addr, metadata !76, metadata !DIExpression()), !dbg !77
  %0 = load %struct.AStruct*, %struct.AStruct** %arg.addr, align 8, !dbg !77
  %coerce.dive = getelementptr inbounds %struct.AStruct, %struct.AStruct* %retval, i32 0, i32 0, !dbg !77
  %1 = load i8, i8* %coerce.dive, align 1, !dbg !77
  ret i8 %1, !dbg !77
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @"?Func_BStruct@@YA?AUBStruct@@AEAU1@@Z"(%struct.BStruct* noalias sret %agg.result, %struct.BStruct* dereferenceable(1) %arg) #0 !dbg !78 {
entry:
  %arg.addr = alloca %struct.BStruct*, align 8
  store %struct.BStruct* %arg, %struct.BStruct** %arg.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.BStruct** %arg.addr, metadata !88, metadata !DIExpression()), !dbg !89
  %0 = load %struct.BStruct*, %struct.BStruct** %arg.addr, align 8, !dbg !89
  ret void, !dbg !89
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i8 @"?Func_AUnion@@YA?ATAUnion@@AEAT1@@Z"(%union.AUnion* dereferenceable(1) %arg) #0 !dbg !90 {
entry:
  %retval = alloca %union.AUnion, align 1
  %arg.addr = alloca %union.AUnion*, align 8
  store %union.AUnion* %arg, %union.AUnion** %arg.addr, align 8
  call void @llvm.dbg.declare(metadata %union.AUnion** %arg.addr, metadata !95, metadata !DIExpression()), !dbg !96
  %0 = load %union.AUnion*, %union.AUnion** %arg.addr, align 8, !dbg !96
  %coerce.dive = getelementptr inbounds %union.AUnion, %union.AUnion* %retval, i32 0, i32 0, !dbg !96
  %1 = load i8, i8* %coerce.dive, align 1, !dbg !96
  ret i8 %1, !dbg !96
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @"?Func_BUnion@@YA?ATBUnion@@AEAT1@@Z"(%union.BUnion* noalias sret %agg.result, %union.BUnion* dereferenceable(1) %arg) #0 !dbg !97 {
entry:
  %arg.addr = alloca %union.BUnion*, align 8
  store %union.BUnion* %arg, %union.BUnion** %arg.addr, align 8
  call void @llvm.dbg.declare(metadata %union.BUnion** %arg.addr, metadata !107, metadata !DIExpression()), !dbg !108
  %0 = load %union.BUnion*, %union.BUnion** %arg.addr, align 8, !dbg !108
  ret void, !dbg !108
}

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 8.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "function-options.cpp", directory: "\5Ctest\5CDebugInfo\5CCOFF", checksumkind: CSK_MD5, checksum: "e73e74ea0bd81174051f0a4746343e00")
!2 = !{}
!3 = !{i32 2, !"CodeView", i32 1}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 2}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = !{!"clang version 8.0.0"}
!8 = distinct !DISubprogram(name: "Func_AClass", linkageName: "?Func_AClass@@YA?AVAClass@@AEAV1@@Z", scope: !9, file: !9, line: 6, type: !10, isLocal: false, isDefinition: true, scopeLine: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!9 = !DIFile(filename: "function-options.cpp", directory: "D:\5Cupstream\5Cllvm\5Ctest\5CDebugInfo\5CCOFF")
!10 = !DISubroutineType(types: !11)
!11 = !{!12, !13}
!12 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "AClass", file: !9, line: 5, size: 8, flags: DIFlagTypePassByValue, elements: !2, identifier: ".?AVAClass@@")
!13 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !12, size: 64)
!14 = !DILocalVariable(name: "arg", arg: 1, scope: !8, file: !9, line: 6, type: !13)
!15 = !DILocation(line: 6, scope: !8)
!16 = distinct !DISubprogram(name: "Func_BClass", linkageName: "?Func_BClass@@YA?AVBClass@@AEAV1@@Z", scope: !9, file: !9, line: 12, type: !17, isLocal: false, isDefinition: true, scopeLine: 12, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!17 = !DISubroutineType(types: !18)
!18 = !{!19, !25}
!19 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "BClass", file: !9, line: 8, size: 8, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !20, identifier: ".?AVBClass@@")
!20 = !{!21}
!21 = !DISubprogram(name: "BClass", scope: !19, file: !9, line: 10, type: !22, isLocal: false, isDefinition: false, scopeLine: 10, flags: DIFlagExplicit | DIFlagPrototyped, isOptimized: false)
!22 = !DISubroutineType(types: !23)
!23 = !{null, !24}
!24 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !19, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!25 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !19, size: 64)
!26 = !DILocalVariable(name: "arg", arg: 1, scope: !16, file: !9, line: 12, type: !25)
!27 = !DILocation(line: 12, scope: !16)
!28 = distinct !DISubprogram(name: "Func_C1Class", linkageName: "?Func_C1Class@@YA?AVC1Class@@AEAV1@@Z", scope: !9, file: !9, line: 18, type: !29, isLocal: false, isDefinition: true, scopeLine: 18, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!29 = !DISubroutineType(types: !30)
!30 = !{!31, !37}
!31 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "C1Class", file: !9, line: 14, size: 8, flags: DIFlagTypePassByValue, elements: !32, identifier: ".?AVC1Class@@")
!32 = !{!33}
!33 = !DISubprogram(name: "C1Class", scope: !31, file: !9, line: 16, type: !34, isLocal: false, isDefinition: false, scopeLine: 16, flags: DIFlagPublic | DIFlagPrototyped, isOptimized: false)
!34 = !DISubroutineType(types: !35)
!35 = !{null, !36}
!36 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !31, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!37 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !31, size: 64)
!38 = !DILocalVariable(name: "arg", arg: 1, scope: !28, file: !9, line: 18, type: !37)
!39 = !DILocation(line: 18, scope: !28)
!40 = distinct !DISubprogram(name: "Func_C2Class", linkageName: "?Func_C2Class@@YA?AVC2Class@@AEAV1@@Z", scope: !9, file: !9, line: 24, type: !41, isLocal: false, isDefinition: true, scopeLine: 24, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!41 = !DISubroutineType(types: !42)
!42 = !{!43, !49}
!43 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "C2Class", file: !9, line: 20, size: 8, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !44, identifier: ".?AVC2Class@@")
!44 = !{!45}
!45 = !DISubprogram(name: "~C2Class", scope: !43, file: !9, line: 22, type: !46, isLocal: false, isDefinition: false, scopeLine: 22, flags: DIFlagPublic | DIFlagPrototyped, isOptimized: false)
!46 = !DISubroutineType(types: !47)
!47 = !{null, !48}
!48 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !43, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!49 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !43, size: 64)
!50 = !DILocalVariable(name: "arg", arg: 1, scope: !40, file: !9, line: 24, type: !49)
!51 = !DILocation(line: 24, scope: !40)
!52 = distinct !DISubprogram(name: "Func_DClass", linkageName: "?Func_DClass@@YA?AVDClass@@AEAV1@@Z", scope: !9, file: !9, line: 27, type: !53, isLocal: false, isDefinition: true, scopeLine: 27, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!53 = !DISubroutineType(types: !54)
!54 = !{!55, !58}
!55 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "DClass", file: !9, line: 26, size: 8, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !56, identifier: ".?AVDClass@@")
!56 = !{!57}
!57 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !55, baseType: !19, flags: DIFlagPublic, extraData: i32 0)
!58 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !55, size: 64)
!59 = !DILocalVariable(name: "arg", arg: 1, scope: !52, file: !9, line: 27, type: !58)
!60 = !DILocation(line: 27, scope: !52)
!61 = distinct !DISubprogram(name: "Func_FClass", linkageName: "?Func_FClass@@YA?AVFClass@@AEAV1@@Z", scope: !9, file: !9, line: 30, type: !62, isLocal: false, isDefinition: true, scopeLine: 30, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!62 = !DISubroutineType(types: !63)
!63 = !{!64, !68}
!64 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "FClass", file: !9, line: 29, size: 8, flags: DIFlagTypePassByValue, elements: !65, identifier: ".?AVFClass@@")
!65 = !{!66}
!66 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !64, file: !9, line: 29, baseType: !67, flags: DIFlagStaticMember)
!67 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!68 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !64, size: 64)
!69 = !DILocalVariable(name: "arg", arg: 1, scope: !61, file: !9, line: 30, type: !68)
!70 = !DILocation(line: 30, scope: !61)
!71 = distinct !DISubprogram(name: "Func_AStruct", linkageName: "?Func_AStruct@@YA?AUAStruct@@AEAU1@@Z", scope: !9, file: !9, line: 33, type: !72, isLocal: false, isDefinition: true, scopeLine: 33, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!72 = !DISubroutineType(types: !73)
!73 = !{!74, !75}
!74 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "AStruct", file: !9, line: 32, size: 8, flags: DIFlagTypePassByValue, elements: !2, identifier: ".?AUAStruct@@")
!75 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !74, size: 64)
!76 = !DILocalVariable(name: "arg", arg: 1, scope: !71, file: !9, line: 33, type: !75)
!77 = !DILocation(line: 33, scope: !71)
!78 = distinct !DISubprogram(name: "Func_BStruct", linkageName: "?Func_BStruct@@YA?AUBStruct@@AEAU1@@Z", scope: !9, file: !9, line: 36, type: !79, isLocal: false, isDefinition: true, scopeLine: 36, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!79 = !DISubroutineType(types: !80)
!80 = !{!81, !87}
!81 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "BStruct", file: !9, line: 35, size: 8, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !82, identifier: ".?AUBStruct@@")
!82 = !{!83}
!83 = !DISubprogram(name: "BStruct", scope: !81, file: !9, line: 35, type: !84, isLocal: false, isDefinition: false, scopeLine: 35, flags: DIFlagPrototyped, isOptimized: false)
!84 = !DISubroutineType(types: !85)
!85 = !{null, !86}
!86 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !81, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!87 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !81, size: 64)
!88 = !DILocalVariable(name: "arg", arg: 1, scope: !78, file: !9, line: 36, type: !87)
!89 = !DILocation(line: 36, scope: !78)
!90 = distinct !DISubprogram(name: "Func_AUnion", linkageName: "?Func_AUnion@@YA?ATAUnion@@AEAT1@@Z", scope: !9, file: !9, line: 39, type: !91, isLocal: false, isDefinition: true, scopeLine: 39, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!91 = !DISubroutineType(types: !92)
!92 = !{!93, !94}
!93 = distinct !DICompositeType(tag: DW_TAG_union_type, name: "AUnion", file: !9, line: 38, size: 8, flags: DIFlagTypePassByValue, elements: !2, identifier: ".?ATAUnion@@")
!94 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !93, size: 64)
!95 = !DILocalVariable(name: "arg", arg: 1, scope: !90, file: !9, line: 39, type: !94)
!96 = !DILocation(line: 39, scope: !90)
!97 = distinct !DISubprogram(name: "Func_BUnion", linkageName: "?Func_BUnion@@YA?ATBUnion@@AEAT1@@Z", scope: !9, file: !9, line: 42, type: !98, isLocal: false, isDefinition: true, scopeLine: 42, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!98 = !DISubroutineType(types: !99)
!99 = !{!100, !106}
!100 = distinct !DICompositeType(tag: DW_TAG_union_type, name: "BUnion", file: !9, line: 41, size: 8, flags: DIFlagTypePassByValue, elements: !101, identifier: ".?ATBUnion@@")
!101 = !{!102}
!102 = !DISubprogram(name: "BUnion", scope: !100, file: !9, line: 41, type: !103, isLocal: false, isDefinition: false, scopeLine: 41, flags: DIFlagPrototyped, isOptimized: false)
!103 = !DISubroutineType(types: !104)
!104 = !{null, !105}
!105 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !100, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!106 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !100, size: 64)
!107 = !DILocalVariable(name: "arg", arg: 1, scope: !97, file: !9, line: 42, type: !106)
!108 = !DILocation(line: 42, scope: !97)
