// RUN: rm -rf %t.cache
// RUN: %clang --target=x86_64-apple-macosx -g -gmodules -Wno-objc-root-class \
// RUN:    -fmodules -fmodules-cache-path=%t.cache \
// RUN:    -c -o %t.o %s -I%S/Inputs
// RUN: lldb-test symbols -dump-clang-ast %t.o | FileCheck --check-prefix CHECK-ANON-S1 %s
// RUN: lldb-test symbols -dump-clang-ast %t.o | FileCheck --check-prefix CHECK-ANON-S2 %s
// RUN: lldb-test symbols -dump-clang-ast %t.o | FileCheck %s
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
// CHECK-ANON-S1: CXXRecordDecl {{.*}} imported in A struct

StructB s3;
// CHECK-ANON-S2: CXXRecordDecl {{.*}} imported in A.B {{.*}}struct
// CHECK-ANON-S2: -FieldDecl {{.*}} in A.B anon_field_b 'int'

Nested s4;
// CHECK-DAG: CXXRecordDecl {{.*}} imported in A struct Nested
// CHECK-DAG: -FieldDecl {{.*}} in A fromb 'StructB'

Enum e1;
// CHECK-DAG: EnumDecl {{.*}} imported in A {{.*}} Enum_e
// FIXME: -EnumConstantDecl {{.*}} imported in A a

@implementation SomeClass {
  int private_ivar;
}
@synthesize number = private_ivar;
@end

SomeClass *obj1;
// RUN: lldb-test symbols -dump-clang-ast -find type --language=ObjC++ \
// RUN:   -compiler-context 'Module:A,Struct:SomeClass' %t.o \
// RUN:   | FileCheck %s --check-prefix=CHECK-OBJC
// CHECK-OBJC: ObjCInterfaceDecl {{.*}} imported in A <undeserialized declarations> SomeClass
// CHECK-OBJC-NEXT: |-ObjCIvarDecl
// CHECK-OBJC-NEXT: |-ObjCMethodDecl 0x[[NUMBER:[0-9a-f]+]]{{.*}} imported in A
// CHECK-OBJC-NEXT: `-ObjCPropertyDecl {{.*}} imported in A number 'int' readonly
// CHECK-OBJC-NEXT:   `-getter ObjCMethod 0x[[NUMBER]] 'number'

// Template specializations are not yet supported, so they lack the ownership info:
Template<int> t2;
// CHECK-DAG: ClassTemplateSpecializationDecl {{.*}} imported in A struct Template

Namespace::InNamespace<int> t3;
// CHECK-DAG: ClassTemplateSpecializationDecl {{.*}} imported in A struct InNamespace

Namespace::AlsoInNamespace<int> t4;
// CHECK-DAG: ClassTemplateSpecializationDecl {{.*}} imported in A.B struct AlsoInNamespace
