; RUN: llc < %s -filetype=obj | llvm-readobj - -codeview | FileCheck %s
;
; Command to generate function-options.ll
; $ clang++ class-options-common.cpp -S -emit-llvm -g -gcodeview -o class-options-common.ll



; // Basically, there are two Property (class-options) expectations on each type:
; // One for class forwarding reference, the other for class definition.
; 
; #define DEFINE_FUNCTION(T) \
;   T Func_##T(T &arg) { return arg; };
; 
; class EmptyClass {}; // Expect: CO = ForwardReference | HasUniqueName
;                      // Expect: CO = HasUniqueName
; DEFINE_FUNCTION(EmptyClass);
; 
; class ExplicitCtorClass { // Expect CO = ForwardReference | HasUniqueName
;                           // Expect CO = HasConstructorOrDestructor | HasUniqueName
; public:
;   explicit ExplicitCtorClass();
; };
; DEFINE_FUNCTION(ExplicitCtorClass);
; 
; class DefaultedCtorClass { // Expect: CO = ForwardReference | HasUniqueName
;                            // Expect: CO = HasUniqueName
; public:
;   DefaultedCtorClass() = default;
; };
; DEFINE_FUNCTION(DefaultedCtorClass);
; 
; class DefaultArgumentCtorClass { // Expect: CO = ForwardReference | HasUniqueName
;                                  // Expect: CO = HasConstructorOrDestructor | HasUniqueName
; public:
;   DefaultArgumentCtorClass(int x = 0);
; };
; DEFINE_FUNCTION(DefaultArgumentCtorClass);
; 
; class UserDtorClass { // Expect: CO = ForwardReference | HasUniqueName
;                       // Expect: CO = HasConstructorOrDestructor| HasUniqueName
; public:
;   ~UserDtorClass() {}
; };
; DEFINE_FUNCTION(UserDtorClass);
; 
; class DefaultedDtorClass { // Expect: CO = ForwardReference | HasUniqueName
;                            // Expect: CO = HasUniqueName
; public:
;   ~DefaultedDtorClass() = default;
; };
; DEFINE_FUNCTION(DefaultedDtorClass);
; 
; class AClass : public ExplicitCtorClass { // Expect: CO = ForwardReference | HasUniqueName
;                                           // Expect: CO = HasConstructorOrDestructor | HasUniqueName
; };
; DEFINE_FUNCTION(AClass);
; 
; class BClass { static int x; }; // Expect: CO = ForwardReference | HasUniqueName
;                                 // Expect: CO = HasUniqueName
; DEFINE_FUNCTION(BClass);
; 
; struct Foo { // Expect: CO = ForwardReference | HasUniqueName
;              // Expect: CO = HasUniqueName
;   Foo() = default;
;   Foo(const Foo &o) = default;
;   int m;
; } f;
; 
; struct Bar { // Expect: CO = ForwardReference | HasUniqueName
;              // Expect: CO = HasConstructorOrDestructor | HasUniqueName
;   int m = 0;
; } b;
; 
; struct AStruct {}; // Expect: CO = ForwardReference | HasUniqueName
;                    // Expect: CO = HasUniqueName
; DEFINE_FUNCTION(AStruct);
; 
; struct BStruct { BStruct(); }; // Expect: CO = ForwardReference | HasUniqueName
;                                // Expect: CO = HasConstructorOrDestructor | HasUniqueName
; DEFINE_FUNCTION(BStruct);
; 
; void S() {
;   struct ComplexStruct { // Expect: CO = ForwardReference | HasUniqueName | Scoped
;                          // Expect: CO = ContainsNestedClass | HasConstructorOrDestructor | HasUniqueName | Scoped
; 
; 
;     struct S {}; // Expect: CO = ForwardReference | HasUniqueName | Nested | Scoped
;                  // Expect: CO = HasUniqueName | Nested | Scoped
; 
;     S s;
;   };
;   ComplexStruct s;
; }
; 
; union AUnion {}; // Expect: CO = ForwardReference | HasUniqueName
;                  // Expect: CO = HasUniqueName | Sealed
; DEFINE_FUNCTION(AUnion);
;
; union BUnion { BUnion() = default; }; // Expect: CO = ForwardReference | HasUniqueName
;                                       // Expect: CO = HasUniqueName | Sealed
; DEFINE_FUNCTION(BUnion);
;
; void U() {
;   union ComplexUnion { // Note clang not yiled 'HasUniqueName' for this type, but MSVC does.
;                        // Expect: CO = ForwardReference | Scoped
;                        // Expect: CO = ContainsNestedClass | Scoped | Sealed
; 
;     union NestedUnion { int x; }; // Note clang not yiled 'HasUniqueName' for this type, but MSVC does.
;                                   // Expect: CO = ForwardReference | Nested | Scoped
;                                   // Expect: CO = Nested | Scoped | Sealed
;     NestedUnion a;
;     int b;
;   };
;   ComplexUnion c;
; }



; CHECK: Format: COFF-x86-64
; CHECK: Arch: x86_64
; CHECK: AddressSize: 64bit
; CHECK: CodeViewTypes [
; CHECK:   Section: .debug$T ({{.*}})
; CHECK:   Magic: 0x4


; CHECK:   Class (0x{{.*}}) {
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
; CHECK:     Name: EmptyClass
; CHECK:     LinkageName: .?AVEmptyClass@@
; CHECK:   }
; CHECK:   Class (0x{{.*}}) {
; CHECK:     TypeLeafKind: LF_CLASS (0x1504)
; CHECK:     MemberCount: 0
; CHECK:     Properties [ (0x200)
; CHECK:       HasUniqueName (0x200)
; CHECK:     ]
; CHECK:     FieldList: <field list> (0x{{.*}})
; CHECK:     DerivedFrom: 0x0
; CHECK:     VShape: 0x0
; CHECK:     SizeOf: 1
; CHECK:     Name: EmptyClass
; CHECK:     LinkageName: .?AVEmptyClass@@
; CHECK:   }


; CHECK:   Class (0x{{.*}}) {
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
; CHECK:     Name: ExplicitCtorClass
; CHECK:     LinkageName: .?AVExplicitCtorClass@@
; CHECK:   }
; CHECK:   Class (0x{{.*}}) {
; CHECK:     TypeLeafKind: LF_CLASS (0x1504)
; CHECK:     MemberCount: 1
; CHECK:     Properties [ (0x202)
; CHECK:       HasConstructorOrDestructor (0x2)
; CHECK:       HasUniqueName (0x200)
; CHECK:     ]
; CHECK:     FieldList: <field list> (0x{{.*}})
; CHECK:     DerivedFrom: 0x0
; CHECK:     VShape: 0x0
; CHECK:     SizeOf: 1
; CHECK:     Name: ExplicitCtorClass
; CHECK:     LinkageName: .?AVExplicitCtorClass@@
; CHECK:   }


; CHECK:   Class (0x{{.*}}) {
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
; CHECK:     Name: DefaultedCtorClass
; CHECK:     LinkageName: .?AVDefaultedCtorClass@@
; CHECK:   }
; CHECK:   Class (0x{{.*}}) {
; CHECK:     TypeLeafKind: LF_CLASS (0x1504)
; CHECK:     MemberCount: 1
; CHECK:     Properties [ (0x200)
; CHECK:       HasUniqueName (0x200)
; CHECK:     ]
; CHECK:     FieldList: <field list> (0x{{.*}})
; CHECK:     DerivedFrom: 0x0
; CHECK:     VShape: 0x0
; CHECK:     SizeOf: 1
; CHECK:     Name: DefaultedCtorClass
; CHECK:     LinkageName: .?AVDefaultedCtorClass@@
; CHECK:   }


; CHECK:   Class (0x{{.*}}) {
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
; CHECK:     Name: DefaultArgumentCtorClass
; CHECK:     LinkageName: .?AVDefaultArgumentCtorClass@@
; CHECK:   }
; CHECK:   Class (0x{{.*}}) {
; CHECK:     TypeLeafKind: LF_CLASS (0x1504)
; CHECK:     MemberCount: 1
; CHECK:     Properties [ (0x202)
; CHECK:       HasConstructorOrDestructor (0x2)
; CHECK:       HasUniqueName (0x200)
; CHECK:     ]
; CHECK:     FieldList: <field list> (0x{{.*}})
; CHECK:     DerivedFrom: 0x0
; CHECK:     VShape: 0x0
; CHECK:     SizeOf: 1
; CHECK:     Name: DefaultArgumentCtorClass
; CHECK:     LinkageName: .?AVDefaultArgumentCtorClass@@
; CHECK:   }


; CHECK:   Class (0x{{.*}}) {
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
; CHECK:     Name: UserDtorClass
; CHECK:     LinkageName: .?AVUserDtorClass@@
; CHECK:   }
; CHECK:   Class (0x{{.*}}) {
; CHECK:     TypeLeafKind: LF_CLASS (0x1504)
; CHECK:     MemberCount: 1
; CHECK:     Properties [ (0x202)
; CHECK:       HasConstructorOrDestructor (0x2)
; CHECK:       HasUniqueName (0x200)
; CHECK:     ]
; CHECK:     FieldList: <field list> (0x{{.*}})
; CHECK:     DerivedFrom: 0x0
; CHECK:     VShape: 0x0
; CHECK:     SizeOf: 1
; CHECK:     Name: UserDtorClass
; CHECK:     LinkageName: .?AVUserDtorClass@@
; CHECK:   }


; CHECK:   Class (0x{{.*}}) {
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
; CHECK:     Name: DefaultedDtorClass
; CHECK:     LinkageName: .?AVDefaultedDtorClass@@
; CHECK:   }
; CHECK:   Class (0x{{.*}}) {
; CHECK:     TypeLeafKind: LF_CLASS (0x1504)
; CHECK:     MemberCount: 1
; CHECK:     Properties [ (0x200)
; CHECK:       HasUniqueName (0x200)
; CHECK:     ]
; CHECK:     FieldList: <field list> (0x{{.*}})
; CHECK:     DerivedFrom: 0x0
; CHECK:     VShape: 0x0
; CHECK:     SizeOf: 1
; CHECK:     Name: DefaultedDtorClass
; CHECK:     LinkageName: .?AVDefaultedDtorClass@@
; CHECK:   }


; CHECK:   Class (0x{{.*}}) {
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
; CHECK:     Name: AClass
; CHECK:     LinkageName: .?AVAClass@@
; CHECK:   }
; CHECK:   Class (0x{{.*}}) {
; CHECK:     TypeLeafKind: LF_CLASS (0x1504)
; CHECK:     MemberCount: 1
; CHECK:     Properties [ (0x202)
; CHECK:       HasConstructorOrDestructor (0x2)
; CHECK:       HasUniqueName (0x200)
; CHECK:     ]
; CHECK:     FieldList: <field list> (0x{{.*}})
; CHECK:     DerivedFrom: 0x0
; CHECK:     VShape: 0x0
; CHECK:     SizeOf: 1
; CHECK:     Name: AClass
; CHECK:     LinkageName: .?AVAClass@@
; CHECK:   }


; CHECK:   Class (0x{{.*}}) {
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
; CHECK:     Name: BClass
; CHECK:     LinkageName: .?AVBClass@@
; CHECK:   }
; CHECK:   Class (0x{{.*}}) {
; CHECK:     TypeLeafKind: LF_CLASS (0x1504)
; CHECK:     MemberCount: 1
; CHECK:     Properties [ (0x200)
; CHECK:       HasUniqueName (0x200)
; CHECK:     ]
; CHECK:     FieldList: <field list> (0x{{.*}})
; CHECK:     DerivedFrom: 0x0
; CHECK:     VShape: 0x0
; CHECK:     SizeOf: 1
; CHECK:     Name: BClass
; CHECK:     LinkageName: .?AVBClass@@
; CHECK:   }


; CHECK:   Struct (0x{{.*}}) {
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
; CHECK:     Name: AStruct
; CHECK:     LinkageName: .?AUAStruct@@
; CHECK:   }
; CHECK:   Struct (0x{{.*}}) {
; CHECK:     TypeLeafKind: LF_STRUCTURE (0x1505)
; CHECK:     MemberCount: 0
; CHECK:     Properties [ (0x200)
; CHECK:       HasUniqueName (0x200)
; CHECK:     ]
; CHECK:     FieldList: <field list> (0x{{.*}})
; CHECK:     DerivedFrom: 0x0
; CHECK:     VShape: 0x0
; CHECK:     SizeOf: 1
; CHECK:     Name: AStruct
; CHECK:     LinkageName: .?AUAStruct@@
; CHECK:   }


; CHECK:   Struct (0x{{.*}}) {
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
; CHECK:     Name: BStruct
; CHECK:     LinkageName: .?AUBStruct@@
; CHECK:   }
; CHECK:   Struct (0x{{.*}}) {
; CHECK:     TypeLeafKind: LF_STRUCTURE (0x1505)
; CHECK:     MemberCount: 1
; CHECK:     Properties [ (0x202)
; CHECK:       HasConstructorOrDestructor (0x2)
; CHECK:       HasUniqueName (0x200)
; CHECK:     ]
; CHECK:     FieldList: <field list> (0x{{.*}})
; CHECK:     DerivedFrom: 0x0
; CHECK:     VShape: 0x0
; CHECK:     SizeOf: 1
; CHECK:     Name: BStruct
; CHECK:     LinkageName: .?AUBStruct@@
; CHECK:   }


; CHECK:   Struct (0x{{.*}}) {
; CHECK:     TypeLeafKind: LF_STRUCTURE (0x1505)
; CHECK:     MemberCount: 0
; CHECK:     Properties [ (0x380)
; CHECK:       ForwardReference (0x80)
; CHECK:       HasUniqueName (0x200)
; CHECK:       Scoped (0x100)
; CHECK:     ]
; CHECK:     FieldList: 0x0
; CHECK:     DerivedFrom: 0x0
; CHECK:     VShape: 0x0
; CHECK:     SizeOf: 0
; CHECK:     Name: S::ComplexStruct
; CHECK:     LinkageName: .?AUComplexStruct@?1??S@@YAXXZ@
; CHECK:   }
; CHECK:   Struct (0x{{.*}}) {
; CHECK:     TypeLeafKind: LF_STRUCTURE (0x1505)
; CHECK:     MemberCount: 0
; CHECK:     Properties [ (0x388)
; CHECK:       ForwardReference (0x80)
; CHECK:       HasUniqueName (0x200)
; CHECK:       Nested (0x8)
; CHECK:       Scoped (0x100)
; CHECK:     ]
; CHECK:     FieldList: 0x0
; CHECK:     DerivedFrom: 0x0
; CHECK:     VShape: 0x0
; CHECK:     SizeOf: 0
; CHECK:     Name: S::ComplexStruct::S
; CHECK:     LinkageName: .?AUS@ComplexStruct@?1??0@YAXXZ@
; CHECK:   }
; CHECK:   Struct (0x{{.*}}) {
; CHECK:     TypeLeafKind: LF_STRUCTURE (0x1505)
; CHECK:     MemberCount: 2
; CHECK:     Properties [ (0x310)
; CHECK:       ContainsNestedClass (0x10)
; CHECK:       HasUniqueName (0x200)
; CHECK:       Scoped (0x100)
; CHECK:     ]
; CHECK:     FieldList: <field list> (0x{{.*}})
; CHECK:     DerivedFrom: 0x0
; CHECK:     VShape: 0x0
; CHECK:     SizeOf: 1
; CHECK:     Name: S::ComplexStruct
; CHECK:     LinkageName: .?AUComplexStruct@?1??S@@YAXXZ@
; CHECK:   }
; CHECK:   Struct (0x{{.*}}) {
; CHECK:     TypeLeafKind: LF_STRUCTURE (0x1505)
; CHECK:     MemberCount: 0
; CHECK:     Properties [ (0x308)
; CHECK:       HasUniqueName (0x200)
; CHECK:       Nested (0x8)
; CHECK:       Scoped (0x100)
; CHECK:     ]
; CHECK:     FieldList: <field list> (0x{{.*}})
; CHECK:     DerivedFrom: 0x0
; CHECK:     VShape: 0x0
; CHECK:     SizeOf: 1
; CHECK:     Name: S::ComplexStruct::S
; CHECK:     LinkageName: .?AUS@ComplexStruct@?1??0@YAXXZ@
; CHECK:   }


; CHECK:   Union (0x1067) {
; CHECK:     TypeLeafKind: LF_UNION (0x1506)
; CHECK:     MemberCount: 0
; CHECK:     Properties [ (0x280)
; CHECK:       ForwardReference (0x80)
; CHECK:       HasUniqueName (0x200)
; CHECK:     ]
; CHECK:     FieldList: 0x0
; CHECK:     SizeOf: 0
; CHECK:     Name: AUnion
; CHECK:     LinkageName: .?ATAUnion@@
; CHECK:   }
; CHECK:   Union (0x106B) {
; CHECK:     TypeLeafKind: LF_UNION (0x1506)
; CHECK:     MemberCount: 0
; CHECK:     Properties [ (0x600)
; CHECK:       HasUniqueName (0x200)
; CHECK:       Sealed (0x400)
; CHECK:     ]
; CHECK:     FieldList: <field list> (0x{{.*}})
; CHECK:     SizeOf: 1
; CHECK:     Name: AUnion
; CHECK:     LinkageName: .?ATAUnion@@
; CHECK:   }


; CHECK:   Union (0x106E) {
; CHECK:     TypeLeafKind: LF_UNION (0x1506)
; CHECK:     MemberCount: 0
; CHECK:     Properties [ (0x280)
; CHECK:       ForwardReference (0x80)
; CHECK:       HasUniqueName (0x200)
; CHECK:     ]
; CHECK:     FieldList: 0x0
; CHECK:     SizeOf: 0
; CHECK:     Name: BUnion
; CHECK:     LinkageName: .?ATBUnion@@
; CHECK:   }
; CHECK:   Union (0x1075) {
; CHECK:     TypeLeafKind: LF_UNION (0x1506)
; CHECK:     MemberCount: 1
; CHECK:     Properties [ (0x600)
; CHECK:       HasUniqueName (0x200)
; CHECK:       Sealed (0x400)
; CHECK:     ]
; CHECK:     FieldList: <field list> (0x{{.*}})
; CHECK:     SizeOf: 1
; CHECK:     Name: BUnion
; CHECK:     LinkageName: .?ATBUnion@@
; CHECK:   }


; CHECK:   Union (0x{{.*}}) {
; CHECK:     TypeLeafKind: LF_UNION (0x1506)
; CHECK:     MemberCount: 0
; CHECK:     Properties [ (0x380)
; CHECK:       ForwardReference (0x80)
; CHECK:       HasUniqueName (0x200)
; CHECK:       Scoped (0x100)
; CHECK:     ]
; CHECK:     FieldList: 0x0
; CHECK:     SizeOf: 0
; CHECK:     Name: U::ComplexUnion
; CHECK:     LinkageName: .?ATComplexUnion@?1??U@@YAXXZ@
; CHECK:   }
; CHECK:   Union (0x{{.*}}) {
; CHECK:     TypeLeafKind: LF_UNION (0x1506)
; CHECK:     MemberCount: 0
; CHECK:     Properties [ (0x388)
; CHECK:       ForwardReference (0x80)
; CHECK:       HasUniqueName (0x200)
; CHECK:       Nested (0x8)
; CHECK:       Scoped (0x100)
; CHECK:     ]
; CHECK:     FieldList: 0x0
; CHECK:     SizeOf: 0
; CHECK:     Name: U::ComplexUnion::NestedUnion
; CHECK:     LinkageName: .?ATNestedUnion@ComplexUnion@?1??U@@YAXXZ@
; CHECK:   }
; CHECK:   Union (0x{{.*}}) {
; CHECK:     TypeLeafKind: LF_UNION (0x1506)
; CHECK:     MemberCount: 3
; CHECK:     Properties [ (0x710)
; CHECK:       ContainsNestedClass (0x10)
; CHECK:       HasUniqueName (0x200)
; CHECK:       Scoped (0x100)
; CHECK:       Sealed (0x400)
; CHECK:     ]
; CHECK:     FieldList: <field list> (0x{{.*}})
; CHECK:     SizeOf: 4
; CHECK:     Name: U::ComplexUnion
; CHECK:     LinkageName: .?ATComplexUnion@?1??U@@YAXXZ@
; CHECK:   }
; CHECK:   Union (0x{{.*}}) {
; CHECK:     TypeLeafKind: LF_UNION (0x1506)
; CHECK:     MemberCount: 1
; CHECK:     Properties [ (0x708)
; CHECK:       HasUniqueName (0x200)
; CHECK:       Nested (0x8)
; CHECK:       Scoped (0x100)
; CHECK:       Sealed (0x400)
; CHECK:     ]
; CHECK:     FieldList: <field list> (0x{{.*}})
; CHECK:     SizeOf: 4
; CHECK:     Name: U::ComplexUnion::NestedUnion
; CHECK:     LinkageName: .?ATNestedUnion@ComplexUnion@?1??U@@YAXXZ@
; CHECK:   }


; CHECK:   Struct (0x{{.*}}) {
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
; CHECK:     Name: Foo
; CHECK:     LinkageName: .?AUFoo@@
; CHECK:   }
; CHECK:   Struct (0x{{.*}}) {
; CHECK:     TypeLeafKind: LF_STRUCTURE (0x1505)
; CHECK:     MemberCount: 3
; CHECK:     Properties [ (0x200)
; CHECK:       HasUniqueName (0x200)
; CHECK:     ]
; CHECK:     FieldList: <field list> (0x{{.*}})
; CHECK:     DerivedFrom: 0x0
; CHECK:     VShape: 0x0
; CHECK:     SizeOf: 4
; CHECK:     Name: Foo
; CHECK:     LinkageName: .?AUFoo@@
; CHECK:   }


; CHECK:   Struct (0x{{.*}}) {
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
; CHECK:     Name: Bar
; CHECK:     LinkageName: .?AUBar@@
; CHECK:   }
; CHECK:   Struct (0x{{.*}}) {
; CHECK:     TypeLeafKind: LF_STRUCTURE (0x1505)
; CHECK:     MemberCount: 1
; CHECK:     Properties [ (0x202)
; CHECK:       HasConstructorOrDestructor (0x2)
; CHECK:       HasUniqueName (0x200)
; CHECK:     ]
; CHECK:     FieldList: <field list> (0x{{.*}})
; CHECK:     DerivedFrom: 0x0
; CHECK:     VShape: 0x0
; CHECK:     SizeOf: 4
; CHECK:     Name: Bar
; CHECK:     LinkageName: .?AUBar@@
; CHECK:   }
; CHECK: ]


; ModuleID = 'class-options-common.cpp'
source_filename = "class-options.cpp"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.15.26729"

%struct.Foo = type { i32 }
%struct.Bar = type { i32 }
%class.EmptyClass = type { i8 }
%class.ExplicitCtorClass = type { i8 }
%class.DefaultedCtorClass = type { i8 }
%class.DefaultArgumentCtorClass = type { i8 }
%class.UserDtorClass = type { i8 }
%class.DefaultedDtorClass = type { i8 }
%class.AClass = type { i8 }
%class.BClass = type { i8 }
%struct.AStruct = type { i8 }
%struct.BStruct = type { i8 }
%struct.ComplexStruct = type { %"struct.S()::ComplexStruct::S" }
%"struct.S()::ComplexStruct::S" = type { i8 }
%union.AUnion = type { i8 }
%union.BUnion = type { i8 }
%union.ComplexUnion = type { %"union.U()::ComplexUnion::NestedUnion" }
%"union.U()::ComplexUnion::NestedUnion" = type { i32 }

@"?f@@3UFoo@@A" = dso_local global %struct.Foo zeroinitializer, align 4, !dbg !0
@"?b@@3UBar@@A" = dso_local global %struct.Bar zeroinitializer, align 4, !dbg !6

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i8 @"?Func_EmptyClass@@YA?AVEmptyClass@@AEAV1@@Z"(%class.EmptyClass* dereferenceable(1) %arg) #0 !dbg !30 {
entry:
  %retval = alloca %class.EmptyClass, align 1
  %arg.addr = alloca %class.EmptyClass*, align 8
  store %class.EmptyClass* %arg, %class.EmptyClass** %arg.addr, align 8
  call void @llvm.dbg.declare(metadata %class.EmptyClass** %arg.addr, metadata !35, metadata !DIExpression()), !dbg !36
  %0 = load %class.EmptyClass*, %class.EmptyClass** %arg.addr, align 8, !dbg !36
  %coerce.dive = getelementptr inbounds %class.EmptyClass, %class.EmptyClass* %retval, i32 0, i32 0, !dbg !36
  %1 = load i8, i8* %coerce.dive, align 1, !dbg !36
  ret i8 %1, !dbg !36
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @"?Func_ExplicitCtorClass@@YA?AVExplicitCtorClass@@AEAV1@@Z"(%class.ExplicitCtorClass* noalias sret %agg.result, %class.ExplicitCtorClass* dereferenceable(1) %arg) #0 !dbg !37 {
entry:
  %arg.addr = alloca %class.ExplicitCtorClass*, align 8
  store %class.ExplicitCtorClass* %arg, %class.ExplicitCtorClass** %arg.addr, align 8
  call void @llvm.dbg.declare(metadata %class.ExplicitCtorClass** %arg.addr, metadata !47, metadata !DIExpression()), !dbg !48
  %0 = load %class.ExplicitCtorClass*, %class.ExplicitCtorClass** %arg.addr, align 8, !dbg !48
  ret void, !dbg !48
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @"?Func_DefaultedCtorClass@@YA?AVDefaultedCtorClass@@AEAV1@@Z"(%class.DefaultedCtorClass* noalias sret %agg.result, %class.DefaultedCtorClass* dereferenceable(1) %arg) #0 !dbg !49 {
entry:
  %arg.addr = alloca %class.DefaultedCtorClass*, align 8
  store %class.DefaultedCtorClass* %arg, %class.DefaultedCtorClass** %arg.addr, align 8
  call void @llvm.dbg.declare(metadata %class.DefaultedCtorClass** %arg.addr, metadata !59, metadata !DIExpression()), !dbg !60
  %0 = load %class.DefaultedCtorClass*, %class.DefaultedCtorClass** %arg.addr, align 8, !dbg !60
  ret void, !dbg !60
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @"?Func_DefaultArgumentCtorClass@@YA?AVDefaultArgumentCtorClass@@AEAV1@@Z"(%class.DefaultArgumentCtorClass* noalias sret %agg.result, %class.DefaultArgumentCtorClass* dereferenceable(1) %arg) #0 !dbg !61 {
entry:
  %arg.addr = alloca %class.DefaultArgumentCtorClass*, align 8
  store %class.DefaultArgumentCtorClass* %arg, %class.DefaultArgumentCtorClass** %arg.addr, align 8
  call void @llvm.dbg.declare(metadata %class.DefaultArgumentCtorClass** %arg.addr, metadata !71, metadata !DIExpression()), !dbg !72
  %0 = load %class.DefaultArgumentCtorClass*, %class.DefaultArgumentCtorClass** %arg.addr, align 8, !dbg !72
  ret void, !dbg !72
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @"?Func_UserDtorClass@@YA?AVUserDtorClass@@AEAV1@@Z"(%class.UserDtorClass* noalias sret %agg.result, %class.UserDtorClass* dereferenceable(1) %arg) #0 !dbg !73 {
entry:
  %arg.addr = alloca %class.UserDtorClass*, align 8
  store %class.UserDtorClass* %arg, %class.UserDtorClass** %arg.addr, align 8
  call void @llvm.dbg.declare(metadata %class.UserDtorClass** %arg.addr, metadata !83, metadata !DIExpression()), !dbg !84
  %0 = load %class.UserDtorClass*, %class.UserDtorClass** %arg.addr, align 8, !dbg !84
  ret void, !dbg !84
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @"?Func_DefaultedDtorClass@@YA?AVDefaultedDtorClass@@AEAV1@@Z"(%class.DefaultedDtorClass* noalias sret %agg.result, %class.DefaultedDtorClass* dereferenceable(1) %arg) #0 !dbg !85 {
entry:
  %arg.addr = alloca %class.DefaultedDtorClass*, align 8
  store %class.DefaultedDtorClass* %arg, %class.DefaultedDtorClass** %arg.addr, align 8
  call void @llvm.dbg.declare(metadata %class.DefaultedDtorClass** %arg.addr, metadata !95, metadata !DIExpression()), !dbg !96
  %0 = load %class.DefaultedDtorClass*, %class.DefaultedDtorClass** %arg.addr, align 8, !dbg !96
  ret void, !dbg !96
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @"?Func_AClass@@YA?AVAClass@@AEAV1@@Z"(%class.AClass* noalias sret %agg.result, %class.AClass* dereferenceable(1) %arg) #0 !dbg !97 {
entry:
  %arg.addr = alloca %class.AClass*, align 8
  store %class.AClass* %arg, %class.AClass** %arg.addr, align 8
  call void @llvm.dbg.declare(metadata %class.AClass** %arg.addr, metadata !104, metadata !DIExpression()), !dbg !105
  %0 = load %class.AClass*, %class.AClass** %arg.addr, align 8, !dbg !105
  ret void, !dbg !105
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i8 @"?Func_BClass@@YA?AVBClass@@AEAV1@@Z"(%class.BClass* dereferenceable(1) %arg) #0 !dbg !106 {
entry:
  %retval = alloca %class.BClass, align 1
  %arg.addr = alloca %class.BClass*, align 8
  store %class.BClass* %arg, %class.BClass** %arg.addr, align 8
  call void @llvm.dbg.declare(metadata %class.BClass** %arg.addr, metadata !113, metadata !DIExpression()), !dbg !114
  %0 = load %class.BClass*, %class.BClass** %arg.addr, align 8, !dbg !114
  %coerce.dive = getelementptr inbounds %class.BClass, %class.BClass* %retval, i32 0, i32 0, !dbg !114
  %1 = load i8, i8* %coerce.dive, align 1, !dbg !114
  ret i8 %1, !dbg !114
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i8 @"?Func_AStruct@@YA?AUAStruct@@AEAU1@@Z"(%struct.AStruct* dereferenceable(1) %arg) #0 !dbg !115 {
entry:
  %retval = alloca %struct.AStruct, align 1
  %arg.addr = alloca %struct.AStruct*, align 8
  store %struct.AStruct* %arg, %struct.AStruct** %arg.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.AStruct** %arg.addr, metadata !120, metadata !DIExpression()), !dbg !121
  %0 = load %struct.AStruct*, %struct.AStruct** %arg.addr, align 8, !dbg !121
  %coerce.dive = getelementptr inbounds %struct.AStruct, %struct.AStruct* %retval, i32 0, i32 0, !dbg !121
  %1 = load i8, i8* %coerce.dive, align 1, !dbg !121
  ret i8 %1, !dbg !121
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @"?Func_BStruct@@YA?AUBStruct@@AEAU1@@Z"(%struct.BStruct* noalias sret %agg.result, %struct.BStruct* dereferenceable(1) %arg) #0 !dbg !122 {
entry:
  %arg.addr = alloca %struct.BStruct*, align 8
  store %struct.BStruct* %arg, %struct.BStruct** %arg.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.BStruct** %arg.addr, metadata !132, metadata !DIExpression()), !dbg !133
  %0 = load %struct.BStruct*, %struct.BStruct** %arg.addr, align 8, !dbg !133
  ret void, !dbg !133
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @"?S@@YAXXZ"() #0 !dbg !134 {
entry:
  %s = alloca %struct.ComplexStruct, align 1
  call void @llvm.dbg.declare(metadata %struct.ComplexStruct* %s, metadata !137, metadata !DIExpression()), !dbg !142
  ret void, !dbg !143
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i8 @"?Func_AUnion@@YA?ATAUnion@@AEAT1@@Z"(%union.AUnion* dereferenceable(1) %arg) #0 !dbg !144 {
entry:
  %retval = alloca %union.AUnion, align 1
  %arg.addr = alloca %union.AUnion*, align 8
  store %union.AUnion* %arg, %union.AUnion** %arg.addr, align 8
  call void @llvm.dbg.declare(metadata %union.AUnion** %arg.addr, metadata !149, metadata !DIExpression()), !dbg !150
  %0 = load %union.AUnion*, %union.AUnion** %arg.addr, align 8, !dbg !150
  %coerce.dive = getelementptr inbounds %union.AUnion, %union.AUnion* %retval, i32 0, i32 0, !dbg !150
  %1 = load i8, i8* %coerce.dive, align 1, !dbg !150
  ret i8 %1, !dbg !150
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @"?Func_BUnion@@YA?ATBUnion@@AEAT1@@Z"(%union.BUnion* noalias sret %agg.result, %union.BUnion* dereferenceable(1) %arg) #0 !dbg !151 {
entry:
  %arg.addr = alloca %union.BUnion*, align 8
  store %union.BUnion* %arg, %union.BUnion** %arg.addr, align 8
  call void @llvm.dbg.declare(metadata %union.BUnion** %arg.addr, metadata !161, metadata !DIExpression()), !dbg !162
  %0 = load %union.BUnion*, %union.BUnion** %arg.addr, align 8, !dbg !162
  ret void, !dbg !162
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @"?U@@YAXXZ"() #0 !dbg !163 {
entry:
  %c = alloca %union.ComplexUnion, align 4
  call void @llvm.dbg.declare(metadata %union.ComplexUnion* %c, metadata !164, metadata !DIExpression()), !dbg !172
  ret void, !dbg !173
}

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!25, !26, !27, !28}
!llvm.ident = !{!29}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "f", linkageName: "?f@@3UFoo@@A", scope: !2, file: !8, line: 60, type: !13, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 8.0.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: None)
!3 = !DIFile(filename: "class-options-common.cpp", directory: "D:\5Cupstream\5Cllvm\5Ctest\5CDebugInfo\5CCOFF", checksumkind: CSK_MD5, checksum: "73d5c55a09899333f27526ae5ea8c878")
!4 = !{}
!5 = !{!0, !6}
!6 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = distinct !DIGlobalVariable(name: "b", linkageName: "?b@@3UBar@@A", scope: !2, file: !8, line: 65, type: !9, isLocal: false, isDefinition: true)
!8 = !DIFile(filename: "class-options.cpp", directory: "D:\5Cupstream\5Cllvm\5Ctest\5CDebugInfo\5CCOFF")
!9 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Bar", file: !8, line: 62, size: 32, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !10, identifier: ".?AUBar@@")
!10 = !{!11}
!11 = !DIDerivedType(tag: DW_TAG_member, name: "m", scope: !9, file: !8, line: 64, baseType: !12, size: 32)
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Foo", file: !8, line: 55, size: 32, flags: DIFlagTypePassByValue, elements: !14, identifier: ".?AUFoo@@")
!14 = !{!15, !16, !20}
!15 = !DIDerivedType(tag: DW_TAG_member, name: "m", scope: !13, file: !8, line: 59, baseType: !12, size: 32)
!16 = !DISubprogram(name: "Foo", scope: !13, file: !8, line: 57, type: !17, isLocal: false, isDefinition: false, scopeLine: 57, flags: DIFlagPrototyped, isOptimized: false)
!17 = !DISubroutineType(types: !18)
!18 = !{null, !19}
!19 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!20 = !DISubprogram(name: "Foo", scope: !13, file: !8, line: 58, type: !21, isLocal: false, isDefinition: false, scopeLine: 58, flags: DIFlagPrototyped, isOptimized: false)
!21 = !DISubroutineType(types: !22)
!22 = !{null, !19, !23}
!23 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !24, size: 64)
!24 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !13)
!25 = !{i32 2, !"CodeView", i32 1}
!26 = !{i32 2, !"Debug Info Version", i32 3}
!27 = !{i32 1, !"wchar_size", i32 2}
!28 = !{i32 7, !"PIC Level", i32 2}
!29 = !{!"clang version 8.0.0 "}
!30 = distinct !DISubprogram(name: "Func_EmptyClass", linkageName: "?Func_EmptyClass@@YA?AVEmptyClass@@AEAV1@@Z", scope: !8, file: !8, line: 9, type: !31, isLocal: false, isDefinition: true, scopeLine: 9, flags: DIFlagPrototyped, isOptimized: false, unit: !2, retainedNodes: !4)
!31 = !DISubroutineType(types: !32)
!32 = !{!33, !34}
!33 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "EmptyClass", file: !8, line: 7, size: 8, flags: DIFlagTypePassByValue, elements: !4, identifier: ".?AVEmptyClass@@")
!34 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !33, size: 64)
!35 = !DILocalVariable(name: "arg", arg: 1, scope: !30, file: !8, line: 9, type: !34)
!36 = !DILocation(line: 9, scope: !30)
!37 = distinct !DISubprogram(name: "Func_ExplicitCtorClass", linkageName: "?Func_ExplicitCtorClass@@YA?AVExplicitCtorClass@@AEAV1@@Z", scope: !8, file: !8, line: 16, type: !38, isLocal: false, isDefinition: true, scopeLine: 16, flags: DIFlagPrototyped, isOptimized: false, unit: !2, retainedNodes: !4)
!38 = !DISubroutineType(types: !39)
!39 = !{!40, !46}
!40 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "ExplicitCtorClass", file: !8, line: 11, size: 8, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !41, identifier: ".?AVExplicitCtorClass@@")
!41 = !{!42}
!42 = !DISubprogram(name: "ExplicitCtorClass", scope: !40, file: !8, line: 14, type: !43, isLocal: false, isDefinition: false, scopeLine: 14, flags: DIFlagPublic | DIFlagExplicit | DIFlagPrototyped, isOptimized: false)
!43 = !DISubroutineType(types: !44)
!44 = !{null, !45}
!45 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !40, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!46 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !40, size: 64)
!47 = !DILocalVariable(name: "arg", arg: 1, scope: !37, file: !8, line: 16, type: !46)
!48 = !DILocation(line: 16, scope: !37)
!49 = distinct !DISubprogram(name: "Func_DefaultedCtorClass", linkageName: "?Func_DefaultedCtorClass@@YA?AVDefaultedCtorClass@@AEAV1@@Z", scope: !8, file: !8, line: 23, type: !50, isLocal: false, isDefinition: true, scopeLine: 23, flags: DIFlagPrototyped, isOptimized: false, unit: !2, retainedNodes: !4)
!50 = !DISubroutineType(types: !51)
!51 = !{!52, !58}
!52 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "DefaultedCtorClass", file: !8, line: 18, size: 8, flags: DIFlagTypePassByValue, elements: !53, identifier: ".?AVDefaultedCtorClass@@")
!53 = !{!54}
!54 = !DISubprogram(name: "DefaultedCtorClass", scope: !52, file: !8, line: 21, type: !55, isLocal: false, isDefinition: false, scopeLine: 21, flags: DIFlagPublic | DIFlagPrototyped, isOptimized: false)
!55 = !DISubroutineType(types: !56)
!56 = !{null, !57}
!57 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !52, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!58 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !52, size: 64)
!59 = !DILocalVariable(name: "arg", arg: 1, scope: !49, file: !8, line: 23, type: !58)
!60 = !DILocation(line: 23, scope: !49)
!61 = distinct !DISubprogram(name: "Func_DefaultArgumentCtorClass", linkageName: "?Func_DefaultArgumentCtorClass@@YA?AVDefaultArgumentCtorClass@@AEAV1@@Z", scope: !8, file: !8, line: 30, type: !62, isLocal: false, isDefinition: true, scopeLine: 30, flags: DIFlagPrototyped, isOptimized: false, unit: !2, retainedNodes: !4)
!62 = !DISubroutineType(types: !63)
!63 = !{!64, !70}
!64 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "DefaultArgumentCtorClass", file: !8, line: 25, size: 8, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !65, identifier: ".?AVDefaultArgumentCtorClass@@")
!65 = !{!66}
!66 = !DISubprogram(name: "DefaultArgumentCtorClass", scope: !64, file: !8, line: 28, type: !67, isLocal: false, isDefinition: false, scopeLine: 28, flags: DIFlagPublic | DIFlagPrototyped, isOptimized: false)
!67 = !DISubroutineType(types: !68)
!68 = !{null, !69, !12}
!69 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !64, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!70 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !64, size: 64)
!71 = !DILocalVariable(name: "arg", arg: 1, scope: !61, file: !8, line: 30, type: !70)
!72 = !DILocation(line: 30, scope: !61)
!73 = distinct !DISubprogram(name: "Func_UserDtorClass", linkageName: "?Func_UserDtorClass@@YA?AVUserDtorClass@@AEAV1@@Z", scope: !8, file: !8, line: 37, type: !74, isLocal: false, isDefinition: true, scopeLine: 37, flags: DIFlagPrototyped, isOptimized: false, unit: !2, retainedNodes: !4)
!74 = !DISubroutineType(types: !75)
!75 = !{!76, !82}
!76 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "UserDtorClass", file: !8, line: 32, size: 8, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !77, identifier: ".?AVUserDtorClass@@")
!77 = !{!78}
!78 = !DISubprogram(name: "~UserDtorClass", scope: !76, file: !8, line: 35, type: !79, isLocal: false, isDefinition: false, scopeLine: 35, flags: DIFlagPublic | DIFlagPrototyped, isOptimized: false)
!79 = !DISubroutineType(types: !80)
!80 = !{null, !81}
!81 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !76, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!82 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !76, size: 64)
!83 = !DILocalVariable(name: "arg", arg: 1, scope: !73, file: !8, line: 37, type: !82)
!84 = !DILocation(line: 37, scope: !73)
!85 = distinct !DISubprogram(name: "Func_DefaultedDtorClass", linkageName: "?Func_DefaultedDtorClass@@YA?AVDefaultedDtorClass@@AEAV1@@Z", scope: !8, file: !8, line: 44, type: !86, isLocal: false, isDefinition: true, scopeLine: 44, flags: DIFlagPrototyped, isOptimized: false, unit: !2, retainedNodes: !4)
!86 = !DISubroutineType(types: !87)
!87 = !{!88, !94}
!88 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "DefaultedDtorClass", file: !8, line: 39, size: 8, flags: DIFlagTypePassByValue, elements: !89, identifier: ".?AVDefaultedDtorClass@@")
!89 = !{!90}
!90 = !DISubprogram(name: "~DefaultedDtorClass", scope: !88, file: !8, line: 42, type: !91, isLocal: false, isDefinition: false, scopeLine: 42, flags: DIFlagPublic | DIFlagPrototyped, isOptimized: false)
!91 = !DISubroutineType(types: !92)
!92 = !{null, !93}
!93 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !88, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!94 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !88, size: 64)
!95 = !DILocalVariable(name: "arg", arg: 1, scope: !85, file: !8, line: 44, type: !94)
!96 = !DILocation(line: 44, scope: !85)
!97 = distinct !DISubprogram(name: "Func_AClass", linkageName: "?Func_AClass@@YA?AVAClass@@AEAV1@@Z", scope: !8, file: !8, line: 49, type: !98, isLocal: false, isDefinition: true, scopeLine: 49, flags: DIFlagPrototyped, isOptimized: false, unit: !2, retainedNodes: !4)
!98 = !DISubroutineType(types: !99)
!99 = !{!100, !103}
!100 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "AClass", file: !8, line: 46, size: 8, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !101, identifier: ".?AVAClass@@")
!101 = !{!102}
!102 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !100, baseType: !40, flags: DIFlagPublic, extraData: i32 0)
!103 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !100, size: 64)
!104 = !DILocalVariable(name: "arg", arg: 1, scope: !97, file: !8, line: 49, type: !103)
!105 = !DILocation(line: 49, scope: !97)
!106 = distinct !DISubprogram(name: "Func_BClass", linkageName: "?Func_BClass@@YA?AVBClass@@AEAV1@@Z", scope: !8, file: !8, line: 53, type: !107, isLocal: false, isDefinition: true, scopeLine: 53, flags: DIFlagPrototyped, isOptimized: false, unit: !2, retainedNodes: !4)
!107 = !DISubroutineType(types: !108)
!108 = !{!109, !112}
!109 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "BClass", file: !8, line: 51, size: 8, flags: DIFlagTypePassByValue, elements: !110, identifier: ".?AVBClass@@")
!110 = !{!111}
!111 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !109, file: !8, line: 51, baseType: !12, flags: DIFlagStaticMember)
!112 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !109, size: 64)
!113 = !DILocalVariable(name: "arg", arg: 1, scope: !106, file: !8, line: 53, type: !112)
!114 = !DILocation(line: 53, scope: !106)
!115 = distinct !DISubprogram(name: "Func_AStruct", linkageName: "?Func_AStruct@@YA?AUAStruct@@AEAU1@@Z", scope: !8, file: !8, line: 69, type: !116, isLocal: false, isDefinition: true, scopeLine: 69, flags: DIFlagPrototyped, isOptimized: false, unit: !2, retainedNodes: !4)
!116 = !DISubroutineType(types: !117)
!117 = !{!118, !119}
!118 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "AStruct", file: !8, line: 67, size: 8, flags: DIFlagTypePassByValue, elements: !4, identifier: ".?AUAStruct@@")
!119 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !118, size: 64)
!120 = !DILocalVariable(name: "arg", arg: 1, scope: !115, file: !8, line: 69, type: !119)
!121 = !DILocation(line: 69, scope: !115)
!122 = distinct !DISubprogram(name: "Func_BStruct", linkageName: "?Func_BStruct@@YA?AUBStruct@@AEAU1@@Z", scope: !8, file: !8, line: 73, type: !123, isLocal: false, isDefinition: true, scopeLine: 73, flags: DIFlagPrototyped, isOptimized: false, unit: !2, retainedNodes: !4)
!123 = !DISubroutineType(types: !124)
!124 = !{!125, !131}
!125 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "BStruct", file: !8, line: 71, size: 8, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !126, identifier: ".?AUBStruct@@")
!126 = !{!127}
!127 = !DISubprogram(name: "BStruct", scope: !125, file: !8, line: 71, type: !128, isLocal: false, isDefinition: false, scopeLine: 71, flags: DIFlagPrototyped, isOptimized: false)
!128 = !DISubroutineType(types: !129)
!129 = !{null, !130}
!130 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !125, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!131 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !125, size: 64)
!132 = !DILocalVariable(name: "arg", arg: 1, scope: !122, file: !8, line: 73, type: !131)
!133 = !DILocation(line: 73, scope: !122)
!134 = distinct !DISubprogram(name: "S", linkageName: "?S@@YAXXZ", scope: !8, file: !8, line: 75, type: !135, isLocal: false, isDefinition: true, scopeLine: 75, flags: DIFlagPrototyped, isOptimized: false, unit: !2, retainedNodes: !4)
!135 = !DISubroutineType(types: !136)
!136 = !{null}
!137 = !DILocalVariable(name: "s", scope: !134, file: !8, line: 85, type: !138)
!138 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "ComplexStruct", scope: !134, file: !8, line: 76, size: 8, flags: DIFlagTypePassByValue, elements: !139, identifier: ".?AUComplexStruct@?1??S@@YAXXZ@")
!139 = !{!140, !141}
!140 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", scope: !138, file: !8, line: 80, size: 8, flags: DIFlagTypePassByValue, elements: !4, identifier: ".?AUS@ComplexStruct@?1??0@YAXXZ@")
!141 = !DIDerivedType(tag: DW_TAG_member, name: "s", scope: !138, file: !8, line: 83, baseType: !140, size: 8)
!142 = !DILocation(line: 85, scope: !134)
!143 = !DILocation(line: 86, scope: !134)
!144 = distinct !DISubprogram(name: "Func_AUnion", linkageName: "?Func_AUnion@@YA?ATAUnion@@AEAT1@@Z", scope: !8, file: !8, line: 90, type: !145, isLocal: false, isDefinition: true, scopeLine: 90, flags: DIFlagPrototyped, isOptimized: false, unit: !2, retainedNodes: !4)
!145 = !DISubroutineType(types: !146)
!146 = !{!147, !148}
!147 = distinct !DICompositeType(tag: DW_TAG_union_type, name: "AUnion", file: !8, line: 88, size: 8, flags: DIFlagTypePassByValue, elements: !4, identifier: ".?ATAUnion@@")
!148 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !147, size: 64)
!149 = !DILocalVariable(name: "arg", arg: 1, scope: !144, file: !8, line: 90, type: !148)
!150 = !DILocation(line: 90, scope: !144)
!151 = distinct !DISubprogram(name: "Func_BUnion", linkageName: "?Func_BUnion@@YA?ATBUnion@@AEAT1@@Z", scope: !8, file: !8, line: 94, type: !152, isLocal: false, isDefinition: true, scopeLine: 94, flags: DIFlagPrototyped, isOptimized: false, unit: !2, retainedNodes: !4)
!152 = !DISubroutineType(types: !153)
!153 = !{!154, !160}
!154 = distinct !DICompositeType(tag: DW_TAG_union_type, name: "BUnion", file: !8, line: 92, size: 8, flags: DIFlagTypePassByValue, elements: !155, identifier: ".?ATBUnion@@")
!155 = !{!156}
!156 = !DISubprogram(name: "BUnion", scope: !154, file: !8, line: 92, type: !157, isLocal: false, isDefinition: false, scopeLine: 92, flags: DIFlagPrototyped, isOptimized: false)
!157 = !DISubroutineType(types: !158)
!158 = !{null, !159}
!159 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !154, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!160 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !154, size: 64)
!161 = !DILocalVariable(name: "arg", arg: 1, scope: !151, file: !8, line: 94, type: !160)
!162 = !DILocation(line: 94, scope: !151)
!163 = distinct !DISubprogram(name: "U", linkageName: "?U@@YAXXZ", scope: !8, file: !8, line: 96, type: !135, isLocal: false, isDefinition: true, scopeLine: 96, flags: DIFlagPrototyped, isOptimized: false, unit: !2, retainedNodes: !4)
!164 = !DILocalVariable(name: "c", scope: !163, file: !8, line: 105, type: !165)
!165 = distinct !DICompositeType(tag: DW_TAG_union_type, name: "ComplexUnion", scope: !163, file: !8, line: 97, size: 32, flags: DIFlagTypePassByValue, elements: !166, identifier: ".?ATComplexUnion@?1??U@@YAXXZ@")
!166 = !{!167, !170, !171}
!167 = distinct !DICompositeType(tag: DW_TAG_union_type, name: "NestedUnion", scope: !165, file: !8, line: 100, size: 32, flags: DIFlagTypePassByValue, elements: !168, identifier: ".?ATNestedUnion@ComplexUnion@?1??U@@YAXXZ@")
!168 = !{!169}
!169 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !167, file: !8, line: 100, baseType: !12, size: 32)
!170 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !165, file: !8, line: 102, baseType: !167, size: 32)
!171 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !165, file: !8, line: 103, baseType: !12, size: 32)
!172 = !DILocation(line: 105, scope: !163)
!173 = !DILocation(line: 106, scope: !163)
