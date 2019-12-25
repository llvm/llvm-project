; RUN: llc < %s -filetype=obj | llvm-readobj - --codeview | FileCheck %s
; RUN: llc < %s | llvm-mc -filetype=obj --triple=x86_64-windows | llvm-readobj - --codeview | FileCheck %s
; RUN: llc < %s | FileCheck %s --check-prefix=ASM

; C++ source to regenerate:
; $ cat t.cpp
; struct A {
;   int a;
;   void f();
; };
; void usevars(int, ...);
; void f(float p1, double p2, long long p3) {
;   int v1 = p3;
;   int *v2 = &v1;
;   const int *v21 = &v1;
;   void *v3 = &v1;
;   int A::*v4 = &A::a;
;   void (A::*v5)() = &A::f;
;   long l1 = 0;
;   long int l2 = 0;
;   unsigned long l3 = 0;
;   unsigned long int l4 = 0;
;   const void *v6 = &v1;
;   usevars(v1, v2, v3, l1, l2, l3, l4);
; }
; void CharTypes() {
;   signed wchar_t w;
;   unsigned short us;
;   char c;
;   unsigned char uc;
;   signed char sc;
;   char16_t c16;
;   char32_t c32;
; }
; $ clang t.cpp -S -emit-llvm -g -gcodeview -o t.ll

; CHECK: CodeViewTypes [
; CHECK:   Section: .debug$T (6)
; CHECK:   Magic: 0x4
; CHECK:   ArgList (0x1000) {
; CHECK:     TypeLeafKind: LF_ARGLIST (0x1201)
; CHECK:     NumArgs: 3
; CHECK:     Arguments [
; CHECK:       ArgType: float (0x40)
; CHECK:       ArgType: double (0x41)
; CHECK:       ArgType: __int64 (0x13)
; CHECK:     ]
; CHECK:   }
; CHECK:   Procedure (0x1001) {
; CHECK:     TypeLeafKind: LF_PROCEDURE (0x1008)
; CHECK:     ReturnType: void (0x3)
; CHECK:     CallingConvention: NearC (0x0)
; CHECK:     FunctionOptions [ (0x0)
; CHECK:     ]
; CHECK:     NumParameters: 3
; CHECK:     ArgListType: (float, double, __int64) (0x1000)
; CHECK:   }
; CHECK:   FuncId (0x1002) {
; CHECK:     TypeLeafKind: LF_FUNC_ID (0x1601)
; CHECK:     ParentScope: 0x0
; CHECK:     FunctionType: void (float, double, __int64) (0x1001)
; CHECK:     Name: f
; CHECK:   }
; CHECK:   Modifier (0x1003) {
; CHECK:     TypeLeafKind: LF_MODIFIER (0x1001)
; CHECK:     ModifiedType: int (0x74)
; CHECK:     Modifiers [ (0x1)
; CHECK:       Const (0x1)
; CHECK:     ]
; CHECK:   }
; CHECK:   Pointer (0x1004) {
; CHECK:     TypeLeafKind: LF_POINTER (0x1002)
; CHECK:     PointeeType: const int (0x1003)
; CHECK:     PtrType: Near64 (0xC)
; CHECK:     PtrMode: Pointer (0x0)
; CHECK:     IsFlat: 0
; CHECK:     IsConst: 0
; CHECK:     IsVolatile: 0
; CHECK:     IsUnaligned: 0
; CHECK:   }
; CHECK:   Struct (0x1005) {
; CHECK:     TypeLeafKind: LF_STRUCTURE (0x1505)
; CHECK:     MemberCount: 0
; CHECK:     Properties [ (0x80)
; CHECK:       ForwardReference (0x80)
; CHECK:     ]
; CHECK:     FieldList: 0x0
; CHECK:     DerivedFrom: 0x0
; CHECK:     VShape: 0x0
; CHECK:     SizeOf: 0
; CHECK:     Name: A
; CHECK:   }
; CHECK:   Pointer (0x1006) {
; CHECK:     TypeLeafKind: LF_POINTER (0x1002)
; CHECK:     PointeeType: int (0x74)
; CHECK:     PtrType: Near64 (0xC)
; CHECK:     PtrMode: PointerToDataMember (0x2)
; CHECK:     IsFlat: 0
; CHECK:     IsConst: 0
; CHECK:     IsVolatile: 0
; CHECK:     IsUnaligned: 0
; CHECK:     ClassType: A (0x1005)
; CHECK:     Representation: GeneralData (0x4)
; CHECK:   }
; CHECK:   Pointer (0x1007) {
; CHECK:     TypeLeafKind: LF_POINTER (0x1002)
; CHECK:     PointeeType: A (0x1005)
; CHECK:     PtrType: Near64 (0xC)
; CHECK:     PtrMode: Pointer (0x0)
; CHECK:     IsFlat: 0
; CHECK:     IsConst: 1
; CHECK:     IsVolatile: 0
; CHECK:     IsUnaligned: 0
; CHECK:   }
; CHECK:   ArgList (0x1008) {
; CHECK:     TypeLeafKind: LF_ARGLIST (0x1201)
; CHECK:     NumArgs: 0
; CHECK:     Arguments [
; CHECK:     ]
; CHECK:   }
; CHECK:   MemberFunction (0x1009) {
; CHECK:     TypeLeafKind: LF_MFUNCTION (0x1009)
; CHECK:     ReturnType: void (0x3)
; CHECK:     ClassType: A (0x1005)
; CHECK:     ThisType: A* const (0x1007)
; CHECK:     CallingConvention: NearC (0x0)
; CHECK:     FunctionOptions [ (0x0)
; CHECK:     ]
; CHECK:     NumParameters: 0
; CHECK:     ArgListType: () (0x1008)
; CHECK:     ThisAdjustment: 0
; CHECK:   }
; CHECK:   FieldList (0x100A) {
; CHECK:     TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK:     DataMember {
; CHECK:       AccessSpecifier: Public (0x3)
; CHECK:       Type: int (0x74)
; CHECK:       FieldOffset: 0x0
; CHECK:       Name: a
; CHECK:     }
; CHECK:     OneMethod {
; CHECK:       AccessSpecifier: Public (0x3)
; CHECK:       Type: void A::() (0x1009)
; CHECK:       Name: A::f
; CHECK:     }
; CHECK:   }
; CHECK:   Struct (0x100B) {
; CHECK:     TypeLeafKind: LF_STRUCTURE (0x1505)
; CHECK:     MemberCount: 2
; CHECK:     Properties [ (0x0)
; CHECK:     ]
; CHECK:     FieldList: <field list> (0x100A)
; CHECK:     DerivedFrom: 0x0
; CHECK:     VShape: 0x0
; CHECK:     SizeOf: 4
; CHECK:     Name: A
; CHECK:   }
; CHECK:   StringId (0x100C) {
; CHECK:     TypeLeafKind: LF_STRING_ID (0x1605)
; CHECK:     Id: 0x0
; CHECK:     StringData: D:\src\llvm\build\t.cpp
; CHECK:   }
; CHECK:   UdtSourceLine (0x100D) {
; CHECK:     TypeLeafKind: LF_UDT_SRC_LINE (0x1606)
; CHECK:     UDT: A (0x100B)
; CHECK:     SourceFile: D:\src\llvm\build\t.cpp (0x100C)
; CHECK:     LineNumber: 1
; CHECK:   }
; CHECK:   Pointer (0x100E) {
; CHECK:     TypeLeafKind: LF_POINTER (0x1002)
; CHECK:     PointeeType: void A::() (0x1009)
; CHECK:     PtrType: Near64 (0xC)
; CHECK:     PtrMode: PointerToMemberFunction (0x3)
; CHECK:     IsFlat: 0
; CHECK:     IsConst: 0
; CHECK:     IsVolatile: 0
; CHECK:     IsUnaligned: 0
; CHECK:     ClassType: A (0x1005)
; CHECK:     Representation: GeneralFunction (0x8)
; CHECK:   }
; CHECK:   Modifier (0x100F) {
; CHECK:     TypeLeafKind: LF_MODIFIER (0x1001)
; CHECK:     ModifiedType: void (0x3)
; CHECK:     Modifiers [ (0x1)
; CHECK:       Const (0x1)
; CHECK:     ]
; CHECK:   }
; CHECK:   Pointer (0x1010) {
; CHECK:     TypeLeafKind: LF_POINTER (0x1002)
; CHECK:     PointeeType: const void (0x100F)
; CHECK:     PtrType: Near64 (0xC)
; CHECK:     PtrMode: Pointer (0x0)
; CHECK:     IsFlat: 0
; CHECK:     IsConst: 0
; CHECK:     IsVolatile: 0
; CHECK:     IsUnaligned: 0
; CHECK:   }
; CHECK:   Procedure (0x1011) {
; CHECK:     TypeLeafKind: LF_PROCEDURE (0x1008)
; CHECK:     ReturnType: void (0x3)
; CHECK:     CallingConvention: NearC (0x0)
; CHECK:     FunctionOptions [ (0x0)
; CHECK:     ]
; CHECK:     NumParameters: 0
; CHECK:     ArgListType: () (0x1008)
; CHECK:   }
; CHECK:   FuncId (0x1012) {
; CHECK:     TypeLeafKind: LF_FUNC_ID (0x1601)
; CHECK:     ParentScope: 0x0
; CHECK:     FunctionType: void () (0x1011)
; CHECK:     Name: CharTypes
; CHECK:   }
; CHECK: ]

; CHECK: CodeViewDebugInfo [
; CHECK:   Subsection [
; CHECK:     SubSectionType: Symbols (0xF1)
; CHECK:     {{.*}}Proc{{.*}}Sym {
; CHECK:       DbgStart: 0x0
; CHECK:       DbgEnd: 0x0
; CHECK:       FunctionType: f (0x1002)
; CHECK:       CodeOffset: ?f@@YAXMN_J@Z+0x0
; CHECK:       Segment: 0x0
; CHECK:       Flags [ (0x0)
; CHECK:       ]
; CHECK:       DisplayName: f
; CHECK:       LinkageName: ?f@@YAXMN_J@Z
; CHECK:     }
; CHECK:     LocalSym {
; CHECK:       Type: float (0x40)
; CHECK:       Flags [ (0x1)
; CHECK:         IsParameter (0x1)
; CHECK:       ]
; CHECK:       VarName: p1
; CHECK:     }
; CHECK:     LocalSym {
; CHECK:       Type: double (0x41)
; CHECK:       Flags [ (0x1)
; CHECK:         IsParameter (0x1)
; CHECK:       ]
; CHECK:       VarName: p2
; CHECK:     }
; CHECK:     LocalSym {
; CHECK:       Type: __int64 (0x13)
; CHECK:       Flags [ (0x1)
; CHECK:         IsParameter (0x1)
; CHECK:       ]
; CHECK:       VarName: p3
; CHECK:     }
; CHECK:     LocalSym {
; CHECK:       Type: int (0x74)
; CHECK:       VarName: v1
; CHECK:     }
; CHECK:     LocalSym {
; CHECK:       Type: int* (0x674)
; CHECK:       VarName: v2
; CHECK:     }
; CHECK:     LocalSym {
; CHECK:       Type: const int* (0x1004)
; CHECK:       VarName: v21
; CHECK:     }
; CHECK:     LocalSym {
; CHECK:       Type: void* (0x603)
; CHECK:       VarName: v3
; CHECK:     }
; CHECK:     LocalSym {
; CHECK:       Type: int A::* (0x1006)
; CHECK:       VarName: v4
; CHECK:     }
; CHECK:     LocalSym {
; CHECK:       Type: void A::() A::* (0x100E)
; CHECK:       VarName: v5
; CHECK:     }
; CHECK:     LocalSym {
; CHECK:       Type: long (0x12)
; CHECK:       VarName: l1
; CHECK:     }
; CHECK:     LocalSym {
; CHECK:       Type: long (0x12)
; CHECK:       VarName: l2
; CHECK:     }
; CHECK:     LocalSym {
; CHECK:       Type: unsigned long (0x22)
; CHECK:       VarName: l3
; CHECK:     }
; CHECK:     LocalSym {
; CHECK:       Type: unsigned long (0x22)
; CHECK:       VarName: l4
; CHECK:     }
; CHECK:     LocalSym {
; CHECK:       Type: const void* (0x1010)
; CHECK:       VarName: v6
; CHECK:     }
; CHECK:     ProcEnd {
; CHECK:     }
; CHECK:   ]
; CHECK:   Subsection [
; CHECK:     {{.*}}Proc{{.*}}Sym {
; CHECK:       Type: CharTypes (0x1012)
; CHECK:       DisplayName: CharTypes
; CHECK:       LinkageName: ?CharTypes@@YAXXZ
; CHECK:     }
; CHECK:     LocalSym {
; CHECK:       Type: wchar_t (0x71)
; CHECK:       Flags [ (0x0)
; CHECK:       ]
; CHECK:       VarName: w
; CHECK:     }
; CHECK:     LocalSym {
; CHECK:       Type: unsigned short (0x21)
; CHECK:       Flags [ (0x0)
; CHECK:       ]
; CHECK:       VarName: us
; CHECK:     }
; CHECK:     LocalSym {
; CHECK:       Type: char (0x70)
; CHECK:       Flags [ (0x0)
; CHECK:       ]
; CHECK:       VarName: c
; CHECK:     }
; CHECK:     LocalSym {
; CHECK:       Type: unsigned char (0x20)
; CHECK:       Flags [ (0x0)
; CHECK:       ]
; CHECK:       VarName: uc
; CHECK:     }
; CHECK:     LocalSym {
; CHECK:       Type: signed char (0x10)
; CHECK:       Flags [ (0x0)
; CHECK:       ]
; CHECK:       VarName: sc
; CHECK:     }
; CHECK:     LocalSym {
; CHECK:       Type: char16_t (0x7A)
; CHECK:       Flags [ (0x0)
; CHECK:       ]
; CHECK:       VarName: c16
; CHECK:     }
; CHECK:     LocalSym {
; CHECK:       Type: char32_t (0x7B)
; CHECK:       Flags [ (0x0)
; CHECK:       ]
; CHECK:       VarName: c32
; CHECK:     }
; CHECK:     ProcEnd {
; CHECK:     }
; CHECK:   ]
; CHECK: ]

; ASM: .section	.debug$T,"dr"
; ASM: .p2align	2
; ASM: .long	4                       # Debug section magic
; ASM: # ArgList (0x1000)
; ASM: .short	0x12                    # Record length
; ASM: .short	0x1201                  # Record kind: LF_ARGLIST
; ASM: .long	0x3                     # NumArgs
; ASM: .long	0x40                    # Argument: float
; ASM: .long	0x41                    # Argument: double
; ASM: .long	0x13                    # Argument: __int64
; ASM: # Procedure (0x1001)
; ASM: .short	0xe                     # Record length
; ASM: .short	0x1008                  # Record kind: LF_PROCEDURE
; ASM: .long	0x3                     # ReturnType: void
; ASM: .byte	0x0                     # CallingConvention: NearC
; ASM: .byte	0x0                     # FunctionOptions
; ASM: .short	0x3                     # NumParameters
; ASM: .long	0x1000                  # ArgListType: (float, double, __int64)
; ASM: # FuncId (0x1002)
; ASM: .short	0xe                     # Record length
; ASM: .short	0x1601                  # Record kind: LF_FUNC_ID
; ASM: .long	0x0                     # ParentScope
; ASM: .long	0x1001                  # FunctionType: void (float, double, __int64)
; ASM: .asciz	"f"                     # Name
; ASM: .byte	242
; ASM: .byte	241
; ASM: # Modifier (0x1003)
; ASM: .short	0xa                     # Record length
; ASM: .short	0x1001                  # Record kind: LF_MODIFIER
; ASM: .long	0x74                    # ModifiedType: int
; ASM: .short	0x1                     # Modifiers ( Const (0x1) )
; ASM: .byte	242
; ASM: .byte	241
; ASM: # Pointer (0x1004)
; ASM: .short	0xa                     # Record length
; ASM: .short	0x1002                  # Record kind: LF_POINTER
; ASM: .long	0x1003                  # PointeeType: const int
; ASM: .long	0x1000c                 # Attrs: [ Type: Near64, Mode: Pointer, SizeOf: 8 ]
; ASM: # Struct (0x1005)
; ASM: .short	0x16                    # Record length
; ASM: .short	0x1505                  # Record kind: LF_STRUCTURE
; ASM: .short	0x0                     # MemberCount
; ASM: .short	0x80                    # Properties ( ForwardReference (0x80) )
; ASM: .long	0x0                     # FieldList
; ASM: .long	0x0                     # DerivedFrom
; ASM: .long	0x0                     # VShape
; ASM: .short	0x0                     # SizeOf
; ASM: .asciz	"A"                     # Name
; ASM: # Pointer (0x1006)
; ASM: .short	0x12                    # Record length
; ASM: .short	0x1002                  # Record kind: LF_POINTER
; ASM: .long	0x74                    # PointeeType: int
; ASM: .long	0x804c                  # Attrs: [ Type: Near64, Mode: PointerToDataMember, SizeOf: 4 ]
; ASM: .long	0x1005                  # ClassType: A
; ASM: .short	0x4                     # Representation: GeneralData
; ASM: .byte	242
; ASM: .byte	241
; ASM: # Pointer (0x1007)
; ASM: .short	0xa                     # Record length
; ASM: .short	0x1002                  # Record kind: LF_POINTER
; ASM: .long	0x1005                  # PointeeType: A
; ASM: .long	0x1040c                 # Attrs: [ Type: Near64, Mode: Pointer, SizeOf: 8, isConst ]
; ASM: # ArgList (0x1008)
; ASM: .short	0x6                     # Record length
; ASM: .short	0x1201                  # Record kind: LF_ARGLIST
; ASM: .long	0x0                     # NumArgs
; ASM: # MemberFunction (0x1009)
; ASM: .short	0x1a                    # Record length
; ASM: .short	0x1009                  # Record kind: LF_MFUNCTION
; ASM: .long	0x3                     # ReturnType: void
; ASM: .long	0x1005                  # ClassType: A
; ASM: .long	0x1007                  # ThisType: A* const
; ASM: .byte	0x0                     # CallingConvention: NearC
; ASM: .byte	0x0                     # FunctionOptions
; ASM: .short	0x0                     # NumParameters
; ASM: .long	0x1008                  # ArgListType: ()
; ASM: .long	0x0                     # ThisAdjustment
; ASM: # FieldList (0x100A)
; ASM: .short	0x1e                    # Record length
; ASM: .short	0x1203                  # Record kind: LF_FIELDLIST
; ASM: .short	0x150d                  # Member kind: DataMember ( LF_MEMBER )
; ASM: .short	0x3                     # Attrs: Public
; ASM: .long	0x74                    # Type: int
; ASM: .short	0x0                     # FieldOffset
; ASM: .asciz	"a"                     # Name
; ASM: .short	0x1511                  # Member kind: OneMethod ( LF_ONEMETHOD )
; ASM: .short	0x3                     # Attrs: Public
; ASM: .long	0x1009                  # Type: void A::()
; ASM: .asciz	"A::f"                  # Name
; ASM: .byte	243
; ASM: .byte	242
; ASM: .byte	241
; ASM: # Struct (0x100B)
; ASM: .short	0x16                    # Record length
; ASM: .short	0x1505                  # Record kind: LF_STRUCTURE
; ASM: .short	0x2                     # MemberCount
; ASM: .short	0x0                     # Properties
; ASM: .long	0x100a                  # FieldList: <field list>
; ASM: .long	0x0                     # DerivedFrom
; ASM: .long	0x0                     # VShape
; ASM: .short	0x4                     # SizeOf
; ASM: .asciz	"A"                     # Name
; ASM: # StringId (0x100C)
; ASM: .short	0x1e                    # Record length
; ASM: .short	0x1605                  # Record kind: LF_STRING_ID
; ASM: .long	0x0                     # Id
; ASM: .asciz	"D:\\src\\llvm\\build\\t.cpp" # StringData
; ASM: # UdtSourceLine (0x100D)
; ASM: .short	0xe                     # Record length
; ASM: .short	0x1606                  # Record kind: LF_UDT_SRC_LINE
; ASM: .long	0x100b                  # UDT: A
; ASM: .long	0x100c                  # SourceFile: D:\src\llvm\build\t.cpp
; ASM: .long	0x1                     # LineNumber
; ASM: # Pointer (0x100E)
; ASM: .short	0x12                    # Record length
; ASM: .short	0x1002                  # Record kind: LF_POINTER
; ASM: .long	0x1009                  # PointeeType: void A::()
; ASM: .long	0x1006c                 # Attrs: [ Type: Near64, Mode: PointerToMemberFunction, SizeOf: 8 ]
; ASM: .long	0x1005                  # ClassType: A
; ASM: .short	0x8                     # Representation: GeneralFunction
; ASM: .byte	242
; ASM: .byte	241
; ASM: # Modifier (0x100F)
; ASM: .short	0xa                     # Record length
; ASM: .short	0x1001                  # Record kind: LF_MODIFIER
; ASM: .long	0x3                     # ModifiedType: void
; ASM: .short	0x1                     # Modifiers ( Const (0x1) )
; ASM: .byte	242
; ASM: .byte	241
; ASM: # Pointer (0x1010)
; ASM: .short	0xa                     # Record length
; ASM: .short	0x1002                  # Record kind: LF_POINTER
; ASM: .long	0x100f                  # PointeeType: const void
; ASM: .long	0x1000c                 # Attrs: [ Type: Near64, Mode: Pointer, SizeOf: 8 ]
; ASM: # Procedure (0x1011)
; ASM: .short	0xe                     # Record length
; ASM: .short	0x1008                  # Record kind: LF_PROCEDURE
; ASM: .long	0x3                     # ReturnType: void
; ASM: .byte	0x0                     # CallingConvention: NearC
; ASM: .byte	0x0                     # FunctionOptions
; ASM: .short	0x0                     # NumParameters
; ASM: .long	0x1008                  # ArgListType: ()
; ASM: # FuncId (0x1012)
; ASM: .short	0x16                    # Record length
; ASM: .short	0x1601                  # Record kind: LF_FUNC_ID
; ASM: .long	0x0                     # ParentScope
; ASM: .long	0x1011                  # FunctionType: void ()
; ASM: .asciz	"CharTypes"             # Name
; ASM: .byte	242
; ASM: .byte	241
; ASM: # StringId (0x1013)
; ASM: .short	0x1a                    # Record length
; ASM: .short	0x1605                  # Record kind: LF_STRING_ID
; ASM: .long	0x0                     # Id
; ASM: .asciz	"D:\\src\\llvm\\build"  # StringData
; ASM: .byte	242
; ASM: .byte	241
; ASM: # StringId (0x1014)
; ASM: .short	0xe                     # Record length
; ASM: .short	0x1605                  # Record kind: LF_STRING_ID
; ASM: .long	0x0                     # Id
; ASM: .asciz	"t.cpp"                 # StringData
; ASM: .byte	242
; ASM: .byte	241
; ASM: # BuildInfo (0x1015)
; ASM: .short	0x1a                    # Record length
; ASM: .short	0x1603                  # Record kind: LF_BUILDINFO
; ASM: .short	0x5                     # NumArgs
; ASM: .long	0x1013                  # Argument: D:\src\llvm\build
; ASM: .long	0x0                     # Argument
; ASM: .long	0x1014                  # Argument: t.cpp
; ASM: .long	0x0                     # Argument
; ASM: .long	0x0                     # Argument
; ASM: .byte	242
; ASM: .byte	241

; ModuleID = 't.cpp'
source_filename = "t.cpp"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.0.23918"

%struct.A = type { i32 }

; Function Attrs: uwtable
define void @"\01?f@@YAXMN_J@Z"(float %p1, double %p2, i64 %p3) #0 !dbg !7 {
entry:
  %p3.addr = alloca i64, align 8
  %p2.addr = alloca double, align 8
  %p1.addr = alloca float, align 4
  %v1 = alloca i32, align 4
  %v2 = alloca i32*, align 8
  %v21 = alloca i32*, align 8
  %v3 = alloca i8*, align 8
  %v4 = alloca i32, align 8
  %v5 = alloca i8*, align 8
  %l1 = alloca i32, align 4
  %l2 = alloca i32, align 4
  %l3 = alloca i32, align 4
  %l4 = alloca i32, align 4
  %v6 = alloca i8*, align 8
  store i64 %p3, i64* %p3.addr, align 8
  call void @llvm.dbg.declare(metadata i64* %p3.addr, metadata !13, metadata !14), !dbg !15
  store double %p2, double* %p2.addr, align 8
  call void @llvm.dbg.declare(metadata double* %p2.addr, metadata !16, metadata !14), !dbg !17
  store float %p1, float* %p1.addr, align 4
  call void @llvm.dbg.declare(metadata float* %p1.addr, metadata !18, metadata !14), !dbg !19
  call void @llvm.dbg.declare(metadata i32* %v1, metadata !20, metadata !14), !dbg !22
  %0 = load i64, i64* %p3.addr, align 8, !dbg !23
  %conv = trunc i64 %0 to i32, !dbg !23
  store i32 %conv, i32* %v1, align 4, !dbg !22
  call void @llvm.dbg.declare(metadata i32** %v2, metadata !24, metadata !14), !dbg !26
  store i32* %v1, i32** %v2, align 8, !dbg !26
  call void @llvm.dbg.declare(metadata i32** %v21, metadata !27, metadata !14), !dbg !30
  store i32* %v1, i32** %v21, align 8, !dbg !30
  call void @llvm.dbg.declare(metadata i8** %v3, metadata !31, metadata !14), !dbg !33
  %1 = bitcast i32* %v1 to i8*, !dbg !34
  store i8* %1, i8** %v3, align 8, !dbg !33
  call void @llvm.dbg.declare(metadata i32* %v4, metadata !35, metadata !14), !dbg !44
  store i32 0, i32* %v4, align 8, !dbg !44
  call void @llvm.dbg.declare(metadata i8** %v5, metadata !45, metadata !14), !dbg !47
  store i8* bitcast (void (%struct.A*)* @"\01?f@A@@QEAAXXZ" to i8*), i8** %v5, align 8, !dbg !47
  call void @llvm.dbg.declare(metadata i32* %l1, metadata !48, metadata !14), !dbg !50
  store i32 0, i32* %l1, align 4, !dbg !50
  call void @llvm.dbg.declare(metadata i32* %l2, metadata !51, metadata !14), !dbg !52
  store i32 0, i32* %l2, align 4, !dbg !52
  call void @llvm.dbg.declare(metadata i32* %l3, metadata !53, metadata !14), !dbg !55
  store i32 0, i32* %l3, align 4, !dbg !55
  call void @llvm.dbg.declare(metadata i32* %l4, metadata !56, metadata !14), !dbg !57
  store i32 0, i32* %l4, align 4, !dbg !57
  call void @llvm.dbg.declare(metadata i8** %v6, metadata !58, metadata !14), !dbg !61
  %2 = bitcast i32* %v1 to i8*, !dbg !62
  store i8* %2, i8** %v6, align 8, !dbg !61
  %3 = load i32, i32* %l4, align 4, !dbg !63
  %4 = load i32, i32* %l3, align 4, !dbg !64
  %5 = load i32, i32* %l2, align 4, !dbg !65
  %6 = load i32, i32* %l1, align 4, !dbg !66
  %7 = load i8*, i8** %v3, align 8, !dbg !67
  %8 = load i32*, i32** %v2, align 8, !dbg !68
  %9 = load i32, i32* %v1, align 4, !dbg !69
  call void (i32, ...) @"\01?usevars@@YAXHZZ"(i32 %9, i32* %8, i8* %7, i32 %6, i32 %5, i32 %4, i32 %3), !dbg !70
  ret void, !dbg !71
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare void @"\01?f@A@@QEAAXXZ"(%struct.A*) #2

declare void @"\01?usevars@@YAXHZZ"(i32, ...) #2

; Function Attrs: nounwind uwtable
define void @"\01?CharTypes@@YAXXZ"() #3 !dbg !72 {
entry:
  %w = alloca i16, align 2
  %us = alloca i16, align 2
  %c = alloca i8, align 1
  %uc = alloca i8, align 1
  %sc = alloca i8, align 1
  %c16 = alloca i16, align 2
  %c32 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i16* %w, metadata !75, metadata !14), !dbg !77
  call void @llvm.dbg.declare(metadata i16* %us, metadata !78, metadata !14), !dbg !80
  call void @llvm.dbg.declare(metadata i8* %c, metadata !81, metadata !14), !dbg !83
  call void @llvm.dbg.declare(metadata i8* %uc, metadata !84, metadata !14), !dbg !86
  call void @llvm.dbg.declare(metadata i8* %sc, metadata !87, metadata !14), !dbg !89
  call void @llvm.dbg.declare(metadata i16* %c16, metadata !90, metadata !14), !dbg !92
  call void @llvm.dbg.declare(metadata i32* %c32, metadata !93, metadata !14), !dbg !95
  ret void, !dbg !96
}

attributes #0 = { uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.9.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "t.cpp", directory: "D:\5Csrc\5Cllvm\5Cbuild")
!2 = !{}
!3 = !{i32 2, !"CodeView", i32 1}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"PIC Level", i32 2}
!6 = !{!"clang version 3.9.0 "}
!7 = distinct !DISubprogram(name: "f", linkageName: "\01?f@@YAXMN_J@Z", scope: !1, file: !1, line: 6, type: !8, isLocal: false, isDefinition: true, scopeLine: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !10, !11, !12}
!10 = !DIBasicType(name: "float", size: 32, align: 32, encoding: DW_ATE_float)
!11 = !DIBasicType(name: "double", size: 64, align: 64, encoding: DW_ATE_float)
!12 = !DIBasicType(name: "long long int", size: 64, align: 64, encoding: DW_ATE_signed)
!13 = !DILocalVariable(name: "p3", arg: 3, scope: !7, file: !1, line: 6, type: !12)
!14 = !DIExpression()
!15 = !DILocation(line: 6, column: 39, scope: !7)
!16 = !DILocalVariable(name: "p2", arg: 2, scope: !7, file: !1, line: 6, type: !11)
!17 = !DILocation(line: 6, column: 25, scope: !7)
!18 = !DILocalVariable(name: "p1", arg: 1, scope: !7, file: !1, line: 6, type: !10)
!19 = !DILocation(line: 6, column: 14, scope: !7)
!20 = !DILocalVariable(name: "v1", scope: !7, file: !1, line: 7, type: !21)
!21 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!22 = !DILocation(line: 7, column: 7, scope: !7)
!23 = !DILocation(line: 7, column: 12, scope: !7)
!24 = !DILocalVariable(name: "v2", scope: !7, file: !1, line: 8, type: !25)
!25 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !21, size: 64, align: 64)
!26 = !DILocation(line: 8, column: 8, scope: !7)
!27 = !DILocalVariable(name: "v21", scope: !7, file: !1, line: 9, type: !28)
!28 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !29, size: 64, align: 64)
!29 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !21)
!30 = !DILocation(line: 9, column: 14, scope: !7)
!31 = !DILocalVariable(name: "v3", scope: !7, file: !1, line: 10, type: !32)
!32 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64, align: 64)
!33 = !DILocation(line: 10, column: 9, scope: !7)
!34 = !DILocation(line: 10, column: 14, scope: !7)
!35 = !DILocalVariable(name: "v4", scope: !7, file: !1, line: 11, type: !36)
!36 = !DIDerivedType(tag: DW_TAG_ptr_to_member_type, baseType: !21, size: 32, extraData: !37)
!37 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "A", file: !1, line: 1, size: 32, align: 32, elements: !38)
!38 = !{!39, !40}
!39 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !37, file: !1, line: 2, baseType: !21, size: 32, align: 32)
!40 = !DISubprogram(name: "A::f", linkageName: "\01?f@A@@QEAAXXZ", scope: !37, file: !1, line: 3, type: !41, isLocal: false, isDefinition: false, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false)
!41 = !DISubroutineType(types: !42)
!42 = !{null, !43}
!43 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !37, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!44 = !DILocation(line: 11, column: 11, scope: !7)
!45 = !DILocalVariable(name: "v5", scope: !7, file: !1, line: 12, type: !46)
!46 = !DIDerivedType(tag: DW_TAG_ptr_to_member_type, baseType: !41, size: 64, extraData: !37)
!47 = !DILocation(line: 12, column: 13, scope: !7)
!48 = !DILocalVariable(name: "l1", scope: !7, file: !1, line: 13, type: !49)
!49 = !DIBasicType(name: "long int", size: 32, align: 32, encoding: DW_ATE_signed)
!50 = !DILocation(line: 13, column: 8, scope: !7)
!51 = !DILocalVariable(name: "l2", scope: !7, file: !1, line: 14, type: !49)
!52 = !DILocation(line: 14, column: 12, scope: !7)
!53 = !DILocalVariable(name: "l3", scope: !7, file: !1, line: 15, type: !54)
!54 = !DIBasicType(name: "long unsigned int", size: 32, align: 32, encoding: DW_ATE_unsigned)
!55 = !DILocation(line: 15, column: 17, scope: !7)
!56 = !DILocalVariable(name: "l4", scope: !7, file: !1, line: 16, type: !54)
!57 = !DILocation(line: 16, column: 21, scope: !7)
!58 = !DILocalVariable(name: "v6", scope: !7, file: !1, line: 17, type: !59)
!59 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !60, size: 64, align: 64)
!60 = !DIDerivedType(tag: DW_TAG_const_type, baseType: null)
!61 = !DILocation(line: 17, column: 15, scope: !7)
!62 = !DILocation(line: 17, column: 20, scope: !7)
!63 = !DILocation(line: 18, column: 35, scope: !7)
!64 = !DILocation(line: 18, column: 31, scope: !7)
!65 = !DILocation(line: 18, column: 27, scope: !7)
!66 = !DILocation(line: 18, column: 23, scope: !7)
!67 = !DILocation(line: 18, column: 19, scope: !7)
!68 = !DILocation(line: 18, column: 15, scope: !7)
!69 = !DILocation(line: 18, column: 11, scope: !7)
!70 = !DILocation(line: 18, column: 3, scope: !7)
!71 = !DILocation(line: 19, column: 1, scope: !7)
!72 = distinct !DISubprogram(name: "CharTypes", linkageName: "\01?CharTypes@@YAXXZ", scope: !1, file: !1, line: 20, type: !73, isLocal: false, isDefinition: true, scopeLine: 20, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!73 = !DISubroutineType(types: !74)
!74 = !{null}
!75 = !DILocalVariable(name: "w", scope: !72, file: !1, line: 21, type: !76)
!76 = !DIBasicType(name: "wchar_t", size: 16, align: 16, encoding: DW_ATE_unsigned)
!77 = !DILocation(line: 21, column: 18, scope: !72)
!78 = !DILocalVariable(name: "us", scope: !72, file: !1, line: 22, type: !79)
!79 = !DIBasicType(name: "unsigned short", size: 16, align: 16, encoding: DW_ATE_unsigned)
!80 = !DILocation(line: 22, column: 18, scope: !72)
!81 = !DILocalVariable(name: "c", scope: !72, file: !1, line: 23, type: !82)
!82 = !DIBasicType(name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!83 = !DILocation(line: 23, column: 8, scope: !72)
!84 = !DILocalVariable(name: "uc", scope: !72, file: !1, line: 24, type: !85)
!85 = !DIBasicType(name: "unsigned char", size: 8, align: 8, encoding: DW_ATE_unsigned_char)
!86 = !DILocation(line: 24, column: 17, scope: !72)
!87 = !DILocalVariable(name: "sc", scope: !72, file: !1, line: 25, type: !88)
!88 = !DIBasicType(name: "signed char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!89 = !DILocation(line: 25, column: 15, scope: !72)
!90 = !DILocalVariable(name: "c16", scope: !72, file: !1, line: 26, type: !91)
!91 = !DIBasicType(name: "char16_t", size: 16, align: 16, encoding: DW_ATE_UTF)
!92 = !DILocation(line: 26, column: 12, scope: !72)
!93 = !DILocalVariable(name: "c32", scope: !72, file: !1, line: 27, type: !94)
!94 = !DIBasicType(name: "char32_t", size: 32, align: 32, encoding: DW_ATE_UTF)
!95 = !DILocation(line: 27, column: 12, scope: !72)
!96 = !DILocation(line: 28, column: 1, scope: !72)
