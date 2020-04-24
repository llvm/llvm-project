// RUN: %clang --target=x86_64-apple-macosx -g -gmodules \
// RUN:    -fmodules -fmodules-cache-path=%t.cache \
// RUN:    -c -o %t.o %s -I%S/Inputs
// RUN: lldb-test symbols -dump-clang-ast %t.o | FileCheck %s
// Verify that the owning module information from DWARF is preserved in the AST.

@import A;

Typedef t1;
// CHECK-DAG: TypedefDecl {{.*}} imported in A Typedef

TopLevelStruct s1;
// CHECK-DAG: CXXRecordDecl {{.*}} imported in A struct TopLevelStruct
// CHECK-DAG: -FieldDecl {{.*}} in A a 'int'

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
// CHECK-DAG: ObjCInterfaceDecl {{.*}} imported in A {{.*}} SomeClass

// Template specializations are not yet supported, so they lack the ownership info:
Template<int> t2;
// CHECK-DAG: ClassTemplateSpecializationDecl {{.*}} struct Template

Namespace::InNamespace<int> t3;
// CHECK-DAG: ClassTemplateSpecializationDecl {{.*}} struct InNamespace

Namespace::AlsoInNamespace<int> t4;
// CHECK-DAG: ClassTemplateSpecializationDecl {{.*}} struct AlsoInNamespace
