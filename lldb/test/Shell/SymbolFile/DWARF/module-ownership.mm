// RUN: %clang --target=x86_64-apple-macosx -g -gmodules \
// RUN:    -fmodules -fmodules-cache-path=%t.cache \
// RUN:    -c -o %t.o %s -I%S/Inputs
// Verify that the owning module information from DWARF is preserved in the AST.

@import A;

Typedef t1;
// RUN: lldb-test symbols -dump-clang-ast -find type --language=ObjC++ \
// RUN:   -compiler-context 'Module:A,Typedef:Typedef' %t.o \
// RUN:   | FileCheck %s --check-prefix=CHECK-TYPEDEF
// CHECK-TYPEDEF: TypedefDecl {{.*}} imported in A Typedef

TopLevelStruct s1;
// RUN: lldb-test symbols -dump-clang-ast -find type --language=ObjC++ \
// RUN:   -compiler-context 'Module:A,Struct:TopLevelStruct' %t.o \
// RUN:   | FileCheck %s --check-prefix=CHECK-TOPLEVELSTRUCT
// CHECK-TOPLEVELSTRUCT: CXXRecordDecl {{.*}} imported in A struct TopLevelStruct
// CHECK-TOPLEVELSTRUCT: -FieldDecl {{.*}} in A a 'int'

Struct s2;
// CHECK-DAG: CXXRecordDecl {{.*}} imported in A struct

StructB s3;
// CHECK-DAG: CXXRecordDecl {{.*}} imported in A.B struct
// CHECK-DAG: -FieldDecl {{.*}} in A.B b 'int'

Nested s4;
// CHECK-DAG: CXXRecordDecl {{.*}} imported in A struct Nested
// CHECK-DAG: -FieldDecl {{.*}} in A fromb 'StructB'

Enum e1;
// CHECK-DAG: EnumDecl {{.*}} imported in A {{.*}} Enum_e
// FIXME: -EnumConstantDecl {{.*}} imported in A a

SomeClass *obj1;
// RUN: lldb-test symbols -dump-clang-ast -find type --language=ObjC++ \
// RUN:   -compiler-context 'Module:A,Struct:SomeClass' %t.o \
// RUN:   | FileCheck %s --check-prefix=CHECK-OBJC
// CHECK-OBJC: ObjCInterfaceDecl {{.*}} imported in A SomeClass
// CHECK-OBJC: |-ObjCPropertyDecl {{.*}} imported in A number 'int' readonly
// CHECK-OBJC: | `-getter ObjCMethod {{.*}} 'number'
// CHECK-OBJC: `-ObjCMethodDecl {{.*}} imported in A implicit - number 'int'

// Template specializations are not yet supported, so they lack the ownership info:
Template<int> t2;
// CHECK-DAG: ClassTemplateSpecializationDecl {{.*}} struct Template

Namespace::InNamespace<int> t3;
// CHECK-DAG: ClassTemplateSpecializationDecl {{.*}} struct InNamespace

Namespace::AlsoInNamespace<int> t4;
// CHECK-DAG: ClassTemplateSpecializationDecl {{.*}} struct AlsoInNamespace
