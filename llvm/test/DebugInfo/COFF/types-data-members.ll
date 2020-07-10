; RUN: llc < %s -filetype=obj | llvm-readobj - --codeview | FileCheck %s
; RUN: llc < %s | llvm-mc -filetype=obj --triple=x86_64-windows | llvm-readobj - --codeview | FileCheck %s
; RUN: llc < %s | FileCheck %s --check-prefix=ASM

; C++ source to regenerate:
; $ cat t.cpp
; struct Struct {
;   int s1;
;   int s2;
;   int s3;
;   static const int sdm = 3;
; };
; union Union {
;   int a;
;   float b;
; };
; class Class {
; public:
;   int pub;
;   struct Nested;
; private:
;   int priv;
; protected:
;   int prot;
; };
; struct DerivedClass : Struct, virtual Class {
;   int d;
; };
; struct Class::Nested {
;   int n;
; };
; void UseTypes() {
;   Struct s;
;   Union u;
;   Class c;
;   DerivedClass dc;
;   Class::Nested n;
; }
; $ clang t.cpp -S -emit-llvm -g -gcodeview -o t.ll

; CHECK: CodeViewTypes [
; CHECK:   Section: .debug$T (8)
; CHECK:   Magic: 0x4
; CHECK:   ArgList (0x1000) {
; CHECK:     TypeLeafKind: LF_ARGLIST (0x1201)
; CHECK:     NumArgs: 0
; CHECK:     Arguments [
; CHECK:     ]
; CHECK:   }
; CHECK:   Procedure (0x1001) {
; CHECK:     TypeLeafKind: LF_PROCEDURE (0x1008)
; CHECK:     ReturnType: void (0x3)
; CHECK:     CallingConvention: NearC (0x0)
; CHECK:     FunctionOptions [ (0x0)
; CHECK:     ]
; CHECK:     NumParameters: 0
; CHECK:     ArgListType: () (0x1000)
; CHECK:   }
; CHECK:   FuncId (0x1002) {
; CHECK:     TypeLeafKind: LF_FUNC_ID (0x1601)
; CHECK:     ParentScope: 0x0
; CHECK:     FunctionType: void () (0x1001)
; CHECK:     Name: UseTypes
; CHECK:   }
; CHECK:   Struct (0x1003) {
; CHECK:     TypeLeafKind: LF_STRUCTURE (0x1505)
; CHECK:     MemberCount: 0
; CHECK:     Properties [ (0x280)
; CHECK:       ForwardReference (0x80)
; CHECK:       HasUniqueName (0x200)
; CHECK:     ]
; CHECK:     FieldList: 0x0
; CHECK:     DerivedFrom: 0x0
; CHECK:     VShape: 0x0
; CHECK:     SizeOf: 0
; CHECK:     Name: Struct
; CHECK:     LinkageName: .?AUStruct@@
; CHECK:   }
; CHECK:   Modifier (0x1004) {
; CHECK:     TypeLeafKind: LF_MODIFIER (0x1001)
; CHECK:     ModifiedType: int (0x74)
; CHECK:     Modifiers [ (0x1)
; CHECK:       Const (0x1)
; CHECK:     ]
; CHECK:   }
; CHECK:   FieldList (0x1005) {
; CHECK:     TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK:     DataMember {
; CHECK:       AccessSpecifier: Public (0x3)
; CHECK:       Type: int (0x74)
; CHECK:       FieldOffset: 0x0
; CHECK:       Name: s1
; CHECK:     }
; CHECK:     DataMember {
; CHECK:       AccessSpecifier: Public (0x3)
; CHECK:       Type: int (0x74)
; CHECK:       FieldOffset: 0x4
; CHECK:       Name: s2
; CHECK:     }
; CHECK:     DataMember {
; CHECK:       AccessSpecifier: Public (0x3)
; CHECK:       Type: int (0x74)
; CHECK:       FieldOffset: 0x8
; CHECK:       Name: s3
; CHECK:     }
; CHECK:     StaticDataMember {
; CHECK:       AccessSpecifier: Public (0x3)
; CHECK:       Type: const int (0x1004)
; CHECK:       Name: sdm
; CHECK:     }
; CHECK:   }
; CHECK:   Struct (0x1006) {
; CHECK:     TypeLeafKind: LF_STRUCTURE (0x1505)
; CHECK:     MemberCount: 4
; CHECK:     Properties [ (0x200)
; CHECK:       HasUniqueName (0x200)
; CHECK:     ]
; CHECK:     FieldList: <field list> (0x1005)
; CHECK:     DerivedFrom: 0x0
; CHECK:     VShape: 0x0
; CHECK:     SizeOf: 12
; CHECK:     Name: Struct
; CHECK:     LinkageName: .?AUStruct@@
; CHECK:   }
; CHECK:   StringId (0x1007) {
; CHECK:     TypeLeafKind: LF_STRING_ID (0x1605)
; CHECK:     Id: 0x0
; CHECK:     StringData: D:\src\llvm\build\t.cpp
; CHECK:   }
; CHECK:   UdtSourceLine (0x1008) {
; CHECK:     TypeLeafKind: LF_UDT_SRC_LINE (0x1606)
; CHECK:     UDT: Struct (0x1006)
; CHECK:     SourceFile: D:\src\llvm\build\t.cpp (0x1007)
; CHECK:     LineNumber: 1
; CHECK:   }
; CHECK:   Union (0x1009) {
; CHECK:     TypeLeafKind: LF_UNION (0x1506)
; CHECK:     MemberCount: 0
; CHECK:     Properties [ (0x280)
; CHECK:       ForwardReference (0x80)
; CHECK:       HasUniqueName (0x200)
; CHECK:     ]
; CHECK:     FieldList: 0x0
; CHECK:     SizeOf: 0
; CHECK:     Name: Union
; CHECK:     LinkageName: .?ATUnion@@
; CHECK:   }
; CHECK:   FieldList (0x100A) {
; CHECK:     TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK:     DataMember {
; CHECK:       AccessSpecifier: Public (0x3)
; CHECK:       Type: int (0x74)
; CHECK:       FieldOffset: 0x0
; CHECK:       Name: a
; CHECK:     }
; CHECK:     DataMember {
; CHECK:       AccessSpecifier: Public (0x3)
; CHECK:       Type: float (0x40)
; CHECK:       FieldOffset: 0x0
; CHECK:       Name: b
; CHECK:     }
; CHECK:   }
; CHECK:   Union (0x100B) {
; CHECK:     TypeLeafKind: LF_UNION (0x1506)
; CHECK:     MemberCount: 2
; CHECK:     Properties [ (0x600)
; CHECK:       HasUniqueName (0x200)
; CHECK:       Sealed (0x400)
; CHECK:     ]
; CHECK:     FieldList: <field list> (0x100A)
; CHECK:     SizeOf: 4
; CHECK:     Name: Union
; CHECK:     LinkageName: .?ATUnion@@
; CHECK:   }
; CHECK:   UdtSourceLine (0x100C) {
; CHECK:     TypeLeafKind: LF_UDT_SRC_LINE (0x1606)
; CHECK:     UDT: Union (0x100B)
; CHECK:     SourceFile: D:\src\llvm\build\t.cpp (0x1007)
; CHECK:     LineNumber: 7
; CHECK:   }
; CHECK:   Class (0x100D) {
; CHECK:     TypeLeafKind: LF_CLASS (0x1504)
; CHECK:     MemberCount: 0
; CHECK:     Properties [ (0x280)
; CHECK:       ForwardReference (0x80)
; CHECK:       HasUniqueName (0x200)
; CHECK:     ]
; CHECK:     FieldList: 0x0
; CHECK:     DerivedFrom: 0x0
; CHECK:     VShape: 0x0
; CHECK:     SizeOf: 0
; CHECK:     Name: Class
; CHECK:     LinkageName: .?AVClass@@
; CHECK:   }
; CHECK:   FieldList (0x100E) {
; CHECK:     TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK:     DataMember {
; CHECK:       AccessSpecifier: Public (0x3)
; CHECK:       Type: int (0x74)
; CHECK:       FieldOffset: 0x0
; CHECK:       Name: pub
; CHECK:     }
; CHECK:     DataMember {
; CHECK:       AccessSpecifier: Private (0x1)
; CHECK:       Type: int (0x74)
; CHECK:       FieldOffset: 0x4
; CHECK:       Name: priv
; CHECK:     }
; CHECK:     DataMember {
; CHECK:       AccessSpecifier: Protected (0x2)
; CHECK:       Type: int (0x74)
; CHECK:       FieldOffset: 0x8
; CHECK:       Name: prot
; CHECK:     }
; CHECK:   }
; CHECK:   Class (0x100F) {
; CHECK:     TypeLeafKind: LF_CLASS (0x1504)
; CHECK:     MemberCount: 3
; CHECK:     Properties [ (0x200)
; CHECK:       HasUniqueName (0x200)
; CHECK:     ]
; CHECK:     FieldList: <field list> (0x100E)
; CHECK:     DerivedFrom: 0x0
; CHECK:     VShape: 0x0
; CHECK:     SizeOf: 12
; CHECK:     Name: Class
; CHECK:     LinkageName: .?AVClass@@
; CHECK:   }
; CHECK:   UdtSourceLine (0x1010) {
; CHECK:     TypeLeafKind: LF_UDT_SRC_LINE (0x1606)
; CHECK:     UDT: Class (0x100F)
; CHECK:     SourceFile: D:\src\llvm\build\t.cpp (0x1007)
; CHECK:     LineNumber: 11
; CHECK:   }
; CHECK:   Struct (0x1011) {
; CHECK:     TypeLeafKind: LF_STRUCTURE (0x1505)
; CHECK:     MemberCount: 0
; CHECK:     Properties [ (0x280)
; CHECK:       ForwardReference (0x80)
; CHECK:       HasUniqueName (0x200)
; CHECK:     ]
; CHECK:     FieldList: 0x0
; CHECK:     DerivedFrom: 0x0
; CHECK:     VShape: 0x0
; CHECK:     SizeOf: 0
; CHECK:     Name: DerivedClass
; CHECK:     LinkageName: .?AUDerivedClass@@
; CHECK:   }
; CHECK:   Pointer (0x1012) {
; CHECK:     TypeLeafKind: LF_POINTER (0x1002)
; CHECK:     PointeeType: const int (0x1004)
; CHECK:     PtrType: Near64 (0xC)
; CHECK:     PtrMode: Pointer (0x0)
; CHECK:     IsFlat: 0
; CHECK:     IsConst: 0
; CHECK:     IsVolatile: 0
; CHECK:     IsUnaligned: 0
; CHECK:     SizeOf: 8
; CHECK:   }
; CHECK:   VFTableShape (0x1013) {
; CHECK:     TypeLeafKind: LF_VTSHAPE (0xA)
; CHECK:     VFEntryCount: 1
; CHECK:   }
; CHECK:   Pointer (0x1014) {
; CHECK:     TypeLeafKind: LF_POINTER (0x1002)
; CHECK:     PointeeType: <vftable 1 methods> (0x1013)
; CHECK:     PtrType: Near64 (0xC)
; CHECK:     PtrMode: Pointer (0x0)
; CHECK:     IsFlat: 0
; CHECK:     IsConst: 0
; CHECK:     IsVolatile: 0
; CHECK:     IsUnaligned: 0
; CHECK:     SizeOf: 8
; CHECK:   }
; CHECK:   FieldList (0x1015) {
; CHECK:     TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK:     BaseClass {
; CHECK:       AccessSpecifier: Public (0x3)
; CHECK:       BaseType: Struct (0x1003)
; CHECK:       BaseOffset: 0x0
; CHECK:     }
; CHECK:     VirtualBaseClass {
; CHECK:       AccessSpecifier: Public (0x3)
; CHECK:       BaseType: Class (0x100D)
; CHECK:       VBPtrType: const int* (0x1012)
; CHECK:       VBPtrOffset: 0x0
; CHECK:       VBTableIndex: 0x1
; CHECK:     }
; CHECK:     VFPtr {
; CHECK:       Type: <vftable 1 methods>* (0x1014)
; CHECK:     }
; CHECK:     DataMember {
; CHECK:       AccessSpecifier: Public (0x3)
; CHECK:       Type: int (0x74)
; CHECK:       FieldOffset: 0x18
; CHECK:       Name: d
; CHECK:     }
; CHECK:   }
; CHECK:   Struct (0x1016) {
; CHECK:     TypeLeafKind: LF_STRUCTURE (0x1505)
; CHECK:     MemberCount: 4
; CHECK:     Properties [ (0x200)
; CHECK:       HasUniqueName (0x200)
; CHECK:     ]
; CHECK:     FieldList: <field list> (0x1015)
; CHECK:     DerivedFrom: 0x0
; CHECK:     VShape: 0x0
; CHECK:     SizeOf: 48
; CHECK:     Name: DerivedClass
; CHECK:     LinkageName: .?AUDerivedClass@@
; CHECK:   }
; CHECK:   UdtSourceLine (0x1017) {
; CHECK:     TypeLeafKind: LF_UDT_SRC_LINE (0x1606)
; CHECK:     UDT: DerivedClass (0x1016)
; CHECK:     SourceFile: D:\src\llvm\build\t.cpp (0x1007)
; CHECK:     LineNumber: 20
; CHECK:   }
; CHECK:   Struct (0x1018) {
; CHECK:     TypeLeafKind: LF_STRUCTURE (0x1505)
; CHECK:     MemberCount: 0
; CHECK:     Properties [ (0x288)
; CHECK:       ForwardReference (0x80)
; CHECK:       HasUniqueName (0x200)
; CHECK:       Nested (0x8)
; CHECK:     ]
; CHECK:     FieldList: 0x0
; CHECK:     DerivedFrom: 0x0
; CHECK:     VShape: 0x0
; CHECK:     SizeOf: 0
; CHECK:     Name: Class::Nested
; CHECK:     LinkageName: .?AUNested@Class@@
; CHECK:   }
; CHECK:   FieldList (0x1019) {
; CHECK:     TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK:     DataMember {
; CHECK:       AccessSpecifier: Public (0x3)
; CHECK:       Type: int (0x74)
; CHECK:       FieldOffset: 0x0
; CHECK:       Name: n
; CHECK:     }
; CHECK:   }
; CHECK:   Struct (0x101A) {
; CHECK:     TypeLeafKind: LF_STRUCTURE (0x1505)
; CHECK:     MemberCount: 1
; CHECK:     Properties [ (0x208)
; CHECK:       HasUniqueName (0x200)
; CHECK:       Nested (0x8)
; CHECK:     ]
; CHECK:     FieldList: <field list> (0x1019)
; CHECK:     DerivedFrom: 0x0
; CHECK:     VShape: 0x0
; CHECK:     SizeOf: 4
; CHECK:     Name: Class::Nested
; CHECK:     LinkageName: .?AUNested@Class@@
; CHECK:   }
; CHECK:   UdtSourceLine (0x101B) {
; CHECK:     TypeLeafKind: LF_UDT_SRC_LINE (0x1606)
; CHECK:     UDT: Class::Nested (0x101A)
; CHECK:     SourceFile: D:\src\llvm\build\t.cpp (0x1007)
; CHECK:     LineNumber: 23
; CHECK:   }
; CHECK:   Pointer (0x101C) {
; CHECK:     TypeLeafKind: LF_POINTER (0x1002)
; CHECK:     PointeeType: DerivedClass (0x1011)
; CHECK:     PtrType: Near64 (0xC)
; CHECK:     PtrMode: Pointer (0x0)
; CHECK:     IsFlat: 0
; CHECK:     IsConst: 1
; CHECK:     IsVolatile: 0
; CHECK:     IsUnaligned: 0
; CHECK:     SizeOf: 8
; CHECK:   }
; CHECK:   MemberFunction (0x101D) {
; CHECK:     TypeLeafKind: LF_MFUNCTION (0x1009)
; CHECK:     ReturnType: void (0x3)
; CHECK:     ClassType: DerivedClass (0x1011)
; CHECK:     ThisType: DerivedClass* const (0x101C)
; CHECK:     CallingConvention: NearC (0x0)
; CHECK:     FunctionOptions [ (0x0)
; CHECK:     ]
; CHECK:     NumParameters: 0
; CHECK:     ArgListType: () (0x1000)
; CHECK:     ThisAdjustment: 0
; CHECK:   }
; CHECK:   MemberFuncId (0x101E) {
; CHECK:     TypeLeafKind: LF_MFUNC_ID (0x1602)
; CHECK:     ClassType: DerivedClass (0x1011)
; CHECK:     FunctionType: void DerivedClass::() (0x101D)
; CHECK:     Name: DerivedClass::DerivedClass
; CHECK:   }
; CHECK: ]

; ASM: .section	.debug$T,"dr"
; ASM: .p2align	2
; ASM: .long	4                       # Debug section magic
; ASM: # ArgList (0x1000)
; ASM: .short	0x6                     # Record length
; ASM: .short	0x1201                  # Record kind: LF_ARGLIST
; ASM: .long	0x0                     # NumArgs
; ASM: # Procedure (0x1001)
; ASM: .short	0xe                     # Record length
; ASM: .short	0x1008                  # Record kind: LF_PROCEDURE
; ASM: .long	0x3                     # ReturnType: void
; ASM: .byte	0x0                     # CallingConvention: NearC
; ASM: .byte	0x0                     # FunctionOptions
; ASM: .short	0x0                     # NumParameters
; ASM: .long	0x1000                  # ArgListType: ()
; ASM: # FuncId (0x1002)
; ASM: .short	0x16                    # Record length
; ASM: .short	0x1601                  # Record kind: LF_FUNC_ID
; ASM: .long	0x0                     # ParentScope
; ASM: .long	0x1001                  # FunctionType: void ()
; ASM: .asciz	"UseTypes"              # Name
; ASM: .byte	243
; ASM: .byte	242
; ASM: .byte	241
; ASM: # Struct (0x1003)
; ASM: .short	0x2a                    # Record length
; ASM: .short	0x1505                  # Record kind: LF_STRUCTURE
; ASM: .short	0x0                     # MemberCount
; ASM: .short	0x280                   # Properties ( ForwardReference (0x80) | HasUniqueName (0x200) )
; ASM: .long	0x0                     # FieldList
; ASM: .long	0x0                     # DerivedFrom
; ASM: .long	0x0                     # VShape
; ASM: .short	0x0                     # SizeOf
; ASM: .asciz	"Struct"                # Name
; ASM: .asciz	".?AUStruct@@"          # LinkageName
; ASM: .byte	242
; ASM: .byte	241
; ASM: # Modifier (0x1004)
; ASM: .short	0xa                     # Record length
; ASM: .short	0x1001                  # Record kind: LF_MODIFIER
; ASM: .long	0x74                    # ModifiedType: int
; ASM: .short	0x1                     # Modifiers ( Const (0x1) )
; ASM: .byte	242
; ASM: .byte	241
; ASM: # FieldList (0x1005)
; ASM: .short	0x3e                    # Record length
; ASM: .short	0x1203                  # Record kind: LF_FIELDLIST
; ASM: .short	0x150d                  # Member kind: DataMember ( LF_MEMBER )
; ASM: .short	0x3                     # Attrs: Public
; ASM: .long	0x74                    # Type: int
; ASM: .short	0x0                     # FieldOffset
; ASM: .asciz	"s1"                    # Name
; ASM: .byte	243
; ASM: .byte	242
; ASM: .byte	241
; ASM: .short	0x150d                  # Member kind: DataMember ( LF_MEMBER )
; ASM: .short	0x3                     # Attrs: Public
; ASM: .long	0x74                    # Type: int
; ASM: .short	0x4                     # FieldOffset
; ASM: .asciz	"s2"                    # Name
; ASM: .byte	243
; ASM: .byte	242
; ASM: .byte	241
; ASM: .short	0x150d                  # Member kind: DataMember ( LF_MEMBER )
; ASM: .short	0x3                     # Attrs: Public
; ASM: .long	0x74                    # Type: int
; ASM: .short	0x8                     # FieldOffset
; ASM: .asciz	"s3"                    # Name
; ASM: .byte	243
; ASM: .byte	242
; ASM: .byte	241
; ASM: .short	0x150e                  # Member kind: StaticDataMember ( LF_STMEMBER )
; ASM: .short	0x3                     # Attrs: Public
; ASM: .long	0x1004                  # Type: const int
; ASM: .asciz	"sdm"                   # Name
; ASM: # Struct (0x1006)
; ASM: .short	0x2a                    # Record length
; ASM: .short	0x1505                  # Record kind: LF_STRUCTURE
; ASM: .short	0x4                     # MemberCount
; ASM: .short	0x200                   # Properties ( HasUniqueName (0x200) )
; ASM: .long	0x1005                  # FieldList: <field list>
; ASM: .long	0x0                     # DerivedFrom
; ASM: .long	0x0                     # VShape
; ASM: .short	0xc                     # SizeOf
; ASM: .asciz	"Struct"                # Name
; ASM: .asciz	".?AUStruct@@"          # LinkageName
; ASM: .byte	242
; ASM: .byte	241
; ASM: # StringId (0x1007)
; ASM: .short	0x1e                    # Record length
; ASM: .short	0x1605                  # Record kind: LF_STRING_ID
; ASM: .long	0x0                     # Id
; ASM: .asciz	"D:\\src\\llvm\\build\\t.cpp" # StringData
; ASM: # UdtSourceLine (0x1008)
; ASM: .short	0xe                     # Record length
; ASM: .short	0x1606                  # Record kind: LF_UDT_SRC_LINE
; ASM: .long	0x1006                  # UDT: Struct
; ASM: .long	0x1007                  # SourceFile: D:\src\llvm\build\t.cpp
; ASM: .long	0x1                     # LineNumber
; ASM: # Union (0x1009)
; ASM: .short	0x1e                    # Record length
; ASM: .short	0x1506                  # Record kind: LF_UNION
; ASM: .short	0x0                     # MemberCount
; ASM: .short	0x280                   # Properties ( ForwardReference (0x80) | HasUniqueName (0x200) )
; ASM: .long	0x0                     # FieldList
; ASM: .short	0x0                     # SizeOf
; ASM: .asciz	"Union"                 # Name
; ASM: .asciz	".?ATUnion@@"           # LinkageName
; ASM: # FieldList (0x100A)
; ASM: .short	0x1a                    # Record length
; ASM: .short	0x1203                  # Record kind: LF_FIELDLIST
; ASM: .short	0x150d                  # Member kind: DataMember ( LF_MEMBER )
; ASM: .short	0x3                     # Attrs: Public
; ASM: .long	0x74                    # Type: int
; ASM: .short	0x0                     # FieldOffset
; ASM: .asciz	"a"                     # Name
; ASM: .short	0x150d                  # Member kind: DataMember ( LF_MEMBER )
; ASM: .short	0x3                     # Attrs: Public
; ASM: .long	0x40                    # Type: float
; ASM: .short	0x0                     # FieldOffset
; ASM: .asciz	"b"                     # Name
; ASM: # Union (0x100B)
; ASM: .short	0x1e                    # Record length
; ASM: .short	0x1506                  # Record kind: LF_UNION
; ASM: .short	0x2                     # MemberCount
; ASM: .short	0x600                   # Properties ( HasUniqueName (0x200) | Sealed (0x400) )
; ASM: .long	0x100a                  # FieldList: <field list>
; ASM: .short	0x4                     # SizeOf
; ASM: .asciz	"Union"                 # Name
; ASM: .asciz	".?ATUnion@@"           # LinkageName
; ASM: # UdtSourceLine (0x100C)
; ASM: .short	0xe                     # Record length
; ASM: .short	0x1606                  # Record kind: LF_UDT_SRC_LINE
; ASM: .long	0x100b                  # UDT: Union
; ASM: .long	0x1007                  # SourceFile: D:\src\llvm\build\t.cpp
; ASM: .long	0x7                     # LineNumber
; ASM: # Class (0x100D)
; ASM: .short	0x26                    # Record length
; ASM: .short	0x1504                  # Record kind: LF_CLASS
; ASM: .short	0x0                     # MemberCount
; ASM: .short	0x280                   # Properties ( ForwardReference (0x80) | HasUniqueName (0x200) )
; ASM: .long	0x0                     # FieldList
; ASM: .long	0x0                     # DerivedFrom
; ASM: .long	0x0                     # VShape
; ASM: .short	0x0                     # SizeOf
; ASM: .asciz	"Class"                 # Name
; ASM: .asciz	".?AVClass@@"           # LinkageName
; ASM: # FieldList (0x100E)
; ASM: .short	0x32                    # Record length
; ASM: .short	0x1203                  # Record kind: LF_FIELDLIST
; ASM: .short	0x150d                  # Member kind: DataMember ( LF_MEMBER )
; ASM: .short	0x3                     # Attrs: Public
; ASM: .long	0x74                    # Type: int
; ASM: .short	0x0                     # FieldOffset
; ASM: .asciz	"pub"                   # Name
; ASM: .byte	242
; ASM: .byte	241
; ASM: .short	0x150d                  # Member kind: DataMember ( LF_MEMBER )
; ASM: .short	0x1                     # Attrs: Private
; ASM: .long	0x74                    # Type: int
; ASM: .short	0x4                     # FieldOffset
; ASM: .asciz	"priv"                  # Name
; ASM: .byte	241
; ASM: .short	0x150d                  # Member kind: DataMember ( LF_MEMBER )
; ASM: .short	0x2                     # Attrs: Protected
; ASM: .long	0x74                    # Type: int
; ASM: .short	0x8                     # FieldOffset
; ASM: .asciz	"prot"                  # Name
; ASM: .byte	241
; ASM: # Class (0x100F)
; ASM: .short	0x26                    # Record length
; ASM: .short	0x1504                  # Record kind: LF_CLASS
; ASM: .short	0x3                     # MemberCount
; ASM: .short	0x200                   # Properties ( HasUniqueName (0x200) )
; ASM: .long	0x100e                  # FieldList: <field list>
; ASM: .long	0x0                     # DerivedFrom
; ASM: .long	0x0                     # VShape
; ASM: .short	0xc                     # SizeOf
; ASM: .asciz	"Class"                 # Name
; ASM: .asciz	".?AVClass@@"           # LinkageName
; ASM: # UdtSourceLine (0x1010)
; ASM: .short	0xe                     # Record length
; ASM: .short	0x1606                  # Record kind: LF_UDT_SRC_LINE
; ASM: .long	0x100f                  # UDT: Class
; ASM: .long	0x1007                  # SourceFile: D:\src\llvm\build\t.cpp
; ASM: .long	0xb                     # LineNumber
; ASM: # Struct (0x1011)
; ASM: .short	0x36                    # Record length
; ASM: .short	0x1505                  # Record kind: LF_STRUCTURE
; ASM: .short	0x0                     # MemberCount
; ASM: .short	0x280                   # Properties ( ForwardReference (0x80) | HasUniqueName (0x200) )
; ASM: .long	0x0                     # FieldList
; ASM: .long	0x0                     # DerivedFrom
; ASM: .long	0x0                     # VShape
; ASM: .short	0x0                     # SizeOf
; ASM: .asciz	"DerivedClass"          # Name
; ASM: .asciz	".?AUDerivedClass@@"    # LinkageName
; ASM: .byte	242
; ASM: .byte	241
; ASM: # Pointer (0x1012)
; ASM: .short	0xa                     # Record length
; ASM: .short	0x1002                  # Record kind: LF_POINTER
; ASM: .long	0x1004                  # PointeeType: const int
; ASM: .long	0x1000c                 # Attrs: [ Type: Near64, Mode: Pointer, SizeOf: 8 ]
; ASM: # VFTableShape (0x1013)
; ASM: .short	0x6                     # Record length
; ASM: .short	0xa                     # Record kind: LF_VTSHAPE
; ASM: .short	0x1                     # VFEntryCount
; ASM: .byte	0x0
; ASM: .byte	241
; ASM: # Pointer (0x1014)
; ASM: .short	0xa                     # Record length
; ASM: .short	0x1002                  # Record kind: LF_POINTER
; ASM: .long	0x1013                  # PointeeType: <vftable 1 methods>
; ASM: .long	0x1000c                 # Attrs: [ Type: Near64, Mode: Pointer, SizeOf: 8 ]
; ASM: # FieldList (0x1015)
; ASM: .short	0x32                    # Record length
; ASM: .short	0x1203                  # Record kind: LF_FIELDLIST
; ASM: .short	0x1400                  # Member kind: BaseClass ( LF_BCLASS )
; ASM: .short	0x3                     # Attrs: Public
; ASM: .long	0x1003                  # BaseType: Struct
; ASM: .short	0x0                     # BaseOffset
; ASM: .byte	242
; ASM: .byte	241
; ASM: .short	0x1401                  # Member kind: VirtualBaseClass ( LF_VBCLASS )
; ASM: .short	0x3                     # Attrs: Public
; ASM: .long	0x100d                  # BaseType: Class
; ASM: .long	0x1012                  # VBPtrType: const int*
; ASM: .short	0x0                     # VBPtrOffset
; ASM: .short	0x1                     # VBTableIndex
; ASM: .short	0x1409                  # Member kind: VFPtr ( LF_VFUNCTAB )
; ASM: .short	0x0                     # Padding
; ASM: .long	0x1014                  # Type: <vftable 1 methods>*
; ASM: .short	0x150d                  # Member kind: DataMember ( LF_MEMBER )
; ASM: .short	0x3                     # Attrs: Public
; ASM: .long	0x74                    # Type: int
; ASM: .short	0x18                    # FieldOffset
; ASM: .asciz	"d"                     # Name
; ASM: # Struct (0x1016)
; ASM: .short	0x36                    # Record length
; ASM: .short	0x1505                  # Record kind: LF_STRUCTURE
; ASM: .short	0x4                     # MemberCount
; ASM: .short	0x200                   # Properties ( HasUniqueName (0x200) )
; ASM: .long	0x1015                  # FieldList: <field list>
; ASM: .long	0x0                     # DerivedFrom
; ASM: .long	0x0                     # VShape
; ASM: .short	0x30                    # SizeOf
; ASM: .asciz	"DerivedClass"          # Name
; ASM: .asciz	".?AUDerivedClass@@"    # LinkageName
; ASM: .byte	242
; ASM: .byte	241
; ASM: # UdtSourceLine (0x1017)
; ASM: .short	0xe                     # Record length
; ASM: .short	0x1606                  # Record kind: LF_UDT_SRC_LINE
; ASM: .long	0x1016                  # UDT: DerivedClass
; ASM: .long	0x1007                  # SourceFile: D:\src\llvm\build\t.cpp
; ASM: .long	0x14                    # LineNumber
; ASM: # Struct (0x1018)
; ASM: .short	0x36                    # Record length
; ASM: .short	0x1505                  # Record kind: LF_STRUCTURE
; ASM: .short	0x0                     # MemberCount
; ASM: .short	0x288                   # Properties ( ForwardReference (0x80) | HasUniqueName (0x200) | Nested (0x8) )
; ASM: .long	0x0                     # FieldList
; ASM: .long	0x0                     # DerivedFrom
; ASM: .long	0x0                     # VShape
; ASM: .short	0x0                     # SizeOf
; ASM: .asciz	"Class::Nested"         # Name
; ASM: .asciz	".?AUNested@Class@@"    # LinkageName
; ASM: .byte	241
; ASM: # FieldList (0x1019)
; ASM: .short	0xe                     # Record length
; ASM: .short	0x1203                  # Record kind: LF_FIELDLIST
; ASM: .short	0x150d                  # Member kind: DataMember ( LF_MEMBER )
; ASM: .short	0x3                     # Attrs: Public
; ASM: .long	0x74                    # Type: int
; ASM: .short	0x0                     # FieldOffset
; ASM: .asciz	"n"                     # Name
; ASM: # Struct (0x101A)
; ASM: .short	0x36                    # Record length
; ASM: .short	0x1505                  # Record kind: LF_STRUCTURE
; ASM: .short	0x1                     # MemberCount
; ASM: .short	0x208                   # Properties ( HasUniqueName (0x200) | Nested (0x8) )
; ASM: .long	0x1019                  # FieldList: <field list>
; ASM: .long	0x0                     # DerivedFrom
; ASM: .long	0x0                     # VShape
; ASM: .short	0x4                     # SizeOf
; ASM: .asciz	"Class::Nested"         # Name
; ASM: .asciz	".?AUNested@Class@@"    # LinkageName
; ASM: .byte	241
; ASM: # UdtSourceLine (0x101B)
; ASM: .short	0xe                     # Record length
; ASM: .short	0x1606                  # Record kind: LF_UDT_SRC_LINE
; ASM: .long	0x101a                  # UDT: Class::Nested
; ASM: .long	0x1007                  # SourceFile: D:\src\llvm\build\t.cpp
; ASM: .long	0x17                    # LineNumber
; ASM: # Pointer (0x101C)
; ASM: .short	0xa                     # Record length
; ASM: .short	0x1002                  # Record kind: LF_POINTER
; ASM: .long	0x1011                  # PointeeType: DerivedClass
; ASM: .long	0x1040c                 # Attrs: [ Type: Near64, Mode: Pointer, SizeOf: 8, isConst ]
; ASM: # MemberFunction (0x101D)
; ASM: .short	0x1a                    # Record length
; ASM: .short	0x1009                  # Record kind: LF_MFUNCTION
; ASM: .long	0x3                     # ReturnType: void
; ASM: .long	0x1011                  # ClassType: DerivedClass
; ASM: .long	0x101c                  # ThisType: DerivedClass* const
; ASM: .byte	0x0                     # CallingConvention: NearC
; ASM: .byte	0x0                     # FunctionOptions
; ASM: .short	0x0                     # NumParameters
; ASM: .long	0x1000                  # ArgListType: ()
; ASM: .long	0x0                     # ThisAdjustment
; ASM: # MemberFuncId (0x101E)
; ASM: .short	0x26                    # Record length
; ASM: .short	0x1602                  # Record kind: LF_MFUNC_ID
; ASM: .long	0x1011                  # ClassType: DerivedClass
; ASM: .long	0x101d                  # FunctionType: void DerivedClass::()
; ASM: .asciz	"DerivedClass::DerivedClass" # Name
; ASM: .byte	241
; ASM: # Pointer (0x101F)
; ASM: .short	0xa                     # Record length
; ASM: .short	0x1002                  # Record kind: LF_POINTER
; ASM: .long	0x1011                  # PointeeType: DerivedClass
; ASM: .long	0x1000c                 # Attrs: [ Type: Near64, Mode: Pointer, SizeOf: 8 ]
; ASM: # StringId (0x1020)
; ASM: .short	0x1a                    # Record length
; ASM: .short	0x1605                  # Record kind: LF_STRING_ID
; ASM: .long	0x0                     # Id
; ASM: .asciz	"D:\\src\\llvm\\build"  # StringData
; ASM: .byte	242
; ASM: .byte	241
; ASM: # StringId (0x1021)
; ASM: .short	0xe                     # Record length
; ASM: .short	0x1605                  # Record kind: LF_STRING_ID
; ASM: .long	0x0                     # Id
; ASM: .asciz	"t.cpp"                 # StringData
; ASM: .byte	242
; ASM: .byte	241
; ASM: # StringId (0x1022)
; ASM: .short	0xa                     # Record length
; ASM: .short	0x1605                  # Record kind: LF_STRING_ID
; ASM: .long	0x0                     # Id
; ASM: .byte    0                       # StringData
; ASM: .byte	243
; ASM: .byte	242
; ASM: .byte	241
; ASM: # BuildInfo (0x1023)
; ASM: .short	0x1a                    # Record length
; ASM: .short	0x1603                  # Record kind: LF_BUILDINFO
; ASM: .short	0x5                     # NumArgs
; ASM: .long	0x1020                  # Argument: D:\src\llvm\build
; ASM: .long	0x0                     # Argument
; ASM: .long	0x1021                  # Argument: t.cpp
; ASM: .long	0x1022                  # Argument
; ASM: .long	0x0                     # Argument
; ASM: .byte	242
; ASM: .byte	241

; ModuleID = 't.cpp'
source_filename = "t.cpp"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.0.23918"

%struct.Struct = type { i32, i32, i32 }
%union.Union = type { i32 }
%class.Class = type { i32, i32, i32 }
%struct.DerivedClass = type { %struct.Struct, i32*, i32, [4 x i8], %class.Class }
%"struct.Class::Nested" = type { i32 }

$"\01??0DerivedClass@@QEAA@XZ" = comdat any

$"\01??_8DerivedClass@@7B@" = comdat any

@"\01??_8DerivedClass@@7B@" = linkonce_odr unnamed_addr constant [2 x i32] [i32 -16, i32 16], comdat

; Function Attrs: nounwind uwtable
define void @"\01?UseTypes@@YAXXZ"() #0 !dbg !7 {
entry:
  %s = alloca %struct.Struct, align 4
  %u = alloca %union.Union, align 4
  %c = alloca %class.Class, align 4
  %dc = alloca %struct.DerivedClass, align 8
  %n = alloca %"struct.Class::Nested", align 4
  call void @llvm.dbg.declare(metadata %struct.Struct* %s, metadata !10, metadata !19), !dbg !20
  call void @llvm.dbg.declare(metadata %union.Union* %u, metadata !21, metadata !19), !dbg !27
  call void @llvm.dbg.declare(metadata %class.Class* %c, metadata !28, metadata !19), !dbg !34
  call void @llvm.dbg.declare(metadata %struct.DerivedClass* %dc, metadata !35, metadata !19), !dbg !46
  %call = call %struct.DerivedClass* @"\01??0DerivedClass@@QEAA@XZ"(%struct.DerivedClass* %dc, i32 1) #3, !dbg !46
  call void @llvm.dbg.declare(metadata %"struct.Class::Nested"* %n, metadata !47, metadata !19), !dbg !51
  ret void, !dbg !52
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr %struct.DerivedClass* @"\01??0DerivedClass@@QEAA@XZ"(%struct.DerivedClass* returned %this, i32 %is_most_derived) unnamed_addr #2 comdat align 2 !dbg !53 {
entry:
  %retval = alloca %struct.DerivedClass*, align 8
  %is_most_derived.addr = alloca i32, align 4
  %this.addr = alloca %struct.DerivedClass*, align 8
  store i32 %is_most_derived, i32* %is_most_derived.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %is_most_derived.addr, metadata !58, metadata !19), !dbg !59
  store %struct.DerivedClass* %this, %struct.DerivedClass** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.DerivedClass** %this.addr, metadata !60, metadata !19), !dbg !59
  %this1 = load %struct.DerivedClass*, %struct.DerivedClass** %this.addr, align 8
  store %struct.DerivedClass* %this1, %struct.DerivedClass** %retval, align 8
  %is_most_derived2 = load i32, i32* %is_most_derived.addr, align 4
  %is_complete_object = icmp ne i32 %is_most_derived2, 0, !dbg !62
  br i1 %is_complete_object, label %ctor.init_vbases, label %ctor.skip_vbases, !dbg !62

ctor.init_vbases:                                 ; preds = %entry
  %this.int8 = bitcast %struct.DerivedClass* %this1 to i8*, !dbg !62
  %0 = getelementptr inbounds i8, i8* %this.int8, i64 16, !dbg !62
  %vbptr.DerivedClass = bitcast i8* %0 to i32**, !dbg !62
  store i32* getelementptr inbounds ([2 x i32], [2 x i32]* @"\01??_8DerivedClass@@7B@", i32 0, i32 0), i32** %vbptr.DerivedClass, align 8, !dbg !62
  %1 = bitcast %struct.DerivedClass* %this1 to i8*, !dbg !62
  %2 = getelementptr inbounds i8, i8* %1, i64 32, !dbg !62
  %3 = bitcast i8* %2 to %class.Class*, !dbg !62
  br label %ctor.skip_vbases, !dbg !62

ctor.skip_vbases:                                 ; preds = %ctor.init_vbases, %entry
  %4 = bitcast %struct.DerivedClass* %this1 to %struct.Struct*, !dbg !62
  %5 = load %struct.DerivedClass*, %struct.DerivedClass** %retval, align 8, !dbg !62
  ret %struct.DerivedClass* %5, !dbg !62
}

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { inlinehint nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind }

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
!7 = distinct !DISubprogram(name: "UseTypes", linkageName: "\01?UseTypes@@YAXXZ", scope: !1, file: !1, line: 26, type: !8, isLocal: false, isDefinition: true, scopeLine: 26, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !DILocalVariable(name: "s", scope: !7, file: !1, line: 27, type: !11)
!11 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Struct", file: !1, line: 1, size: 96, align: 32, elements: !12, identifier: ".?AUStruct@@")
!12 = !{!13, !15, !16, !17}
!13 = !DIDerivedType(tag: DW_TAG_member, name: "s1", scope: !11, file: !1, line: 2, baseType: !14, size: 32, align: 32)
!14 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!15 = !DIDerivedType(tag: DW_TAG_member, name: "s2", scope: !11, file: !1, line: 3, baseType: !14, size: 32, align: 32, offset: 32)
!16 = !DIDerivedType(tag: DW_TAG_member, name: "s3", scope: !11, file: !1, line: 4, baseType: !14, size: 32, align: 32, offset: 64)
!17 = !DIDerivedType(tag: DW_TAG_member, name: "sdm", scope: !11, file: !1, line: 5, baseType: !18, flags: DIFlagStaticMember, extraData: i32 3)
!18 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !14)
!19 = !DIExpression()
!20 = !DILocation(line: 27, column: 10, scope: !7)
!21 = !DILocalVariable(name: "u", scope: !7, file: !1, line: 28, type: !22)
!22 = distinct !DICompositeType(tag: DW_TAG_union_type, name: "Union", file: !1, line: 7, size: 32, align: 32, elements: !23, identifier: ".?ATUnion@@")
!23 = !{!24, !25}
!24 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !22, file: !1, line: 8, baseType: !14, size: 32, align: 32)
!25 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !22, file: !1, line: 9, baseType: !26, size: 32, align: 32)
!26 = !DIBasicType(name: "float", size: 32, align: 32, encoding: DW_ATE_float)
!27 = !DILocation(line: 28, column: 9, scope: !7)
!28 = !DILocalVariable(name: "c", scope: !7, file: !1, line: 29, type: !29)
!29 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "Class", file: !1, line: 11, size: 96, align: 32, elements: !30, identifier: ".?AVClass@@")
!30 = !{!31, !32, !33}
!31 = !DIDerivedType(tag: DW_TAG_member, name: "pub", scope: !29, file: !1, line: 13, baseType: !14, size: 32, align: 32, flags: DIFlagPublic)
!32 = !DIDerivedType(tag: DW_TAG_member, name: "priv", scope: !29, file: !1, line: 16, baseType: !14, size: 32, align: 32, offset: 32)
!33 = !DIDerivedType(tag: DW_TAG_member, name: "prot", scope: !29, file: !1, line: 18, baseType: !14, size: 32, align: 32, offset: 64, flags: DIFlagProtected)
!34 = !DILocation(line: 29, column: 9, scope: !7)
!35 = !DILocalVariable(name: "dc", scope: !7, file: !1, line: 30, type: !36)
!36 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "DerivedClass", file: !1, line: 20, size: 384, align: 64, elements: !37, vtableHolder: !36, identifier: ".?AUDerivedClass@@")
!37 = !{!38, !39, !40, !45}
!38 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !36, baseType: !11)
!39 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !36, baseType: !29, offset: 4, flags: DIFlagVirtual)
!40 = !DIDerivedType(tag: DW_TAG_member, name: "_vptr$DerivedClass", scope: !1, file: !1, baseType: !41, size: 64, flags: DIFlagArtificial)
!41 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !42, size: 64)
!42 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "__vtbl_ptr_type", baseType: !43, size: 64)
!43 = !DISubroutineType(types: !44)
!44 = !{!14}
!45 = !DIDerivedType(tag: DW_TAG_member, name: "d", scope: !36, file: !1, line: 21, baseType: !14, size: 32, align: 32, offset: 192)
!46 = !DILocation(line: 30, column: 16, scope: !7)
!47 = !DILocalVariable(name: "n", scope: !7, file: !1, line: 31, type: !48)
!48 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Nested", scope: !29, file: !1, line: 23, size: 32, align: 32, elements: !49, identifier: ".?AUNested@Class@@")
!49 = !{!50}
!50 = !DIDerivedType(tag: DW_TAG_member, name: "n", scope: !48, file: !1, line: 24, baseType: !14, size: 32, align: 32)
!51 = !DILocation(line: 31, column: 17, scope: !7)
!52 = !DILocation(line: 32, column: 1, scope: !7)
!53 = distinct !DISubprogram(name: "DerivedClass::DerivedClass", linkageName: "\01??0DerivedClass@@QEAA@XZ", scope: !36, file: !1, line: 20, type: !54, isLocal: false, isDefinition: true, scopeLine: 20, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false, unit: !0, declaration: !57, retainedNodes: !2)
!54 = !DISubroutineType(types: !55)
!55 = !{null, !56}
!56 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !36, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!57 = !DISubprogram(name: "DerivedClass::DerivedClass", scope: !36, type: !54, isLocal: false, isDefinition: false, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false)
!58 = !DILocalVariable(name: "is_most_derived", arg: 2, scope: !53, type: !14, flags: DIFlagArtificial)
!59 = !DILocation(line: 0, scope: !53)
!60 = !DILocalVariable(name: "this", arg: 1, scope: !53, type: !61, flags: DIFlagArtificial | DIFlagObjectPointer)
!61 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !36, size: 64, align: 64)
!62 = !DILocation(line: 20, column: 8, scope: !53)
