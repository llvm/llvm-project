// Completing an Objective-C interface from DWARF must not go through the
// Microsoft C++ ABI inheritance model code, which only applies to
// CXXRecordDecls. Before this was guarded, every Objective-C type completion
// for a target using the Microsoft C++ ABI crashed LLDB.
// REQUIRES: lld, x86

// RUN: %clang --target=x86_64-pc-windows-msvc -gdwarf -c -o %t.obj -- %s
// RUN: lld-link -debug:dwarf -nodefaultlib -force:unresolved -entry:main \
// RUN:     -out:%t.exe -- %t.obj
// RUN: lldb-test symbols -dump-clang-ast %t.exe | FileCheck %s

// CHECK: ObjCInterfaceDecl {{.*}} Base
// CHECK-NEXT: ObjCIvarDecl {{.*}} base_ivar 'int'
// CHECK: ObjCInterfaceDecl {{.*}} Derived
// CHECK-NEXT: super ObjCInterface {{.*}} 'Base'
// CHECK-NEXT: ObjCIvarDecl {{.*}} derived_ivar 'int'

__attribute__((objc_root_class))
@interface Base {
  int base_ivar;
}
@end

@implementation Base
@end

@interface Derived : Base {
  int derived_ivar;
}
@end

@implementation Derived
@end

int main(void) {
  Derived *d = 0;
  return (int)(__SIZE_TYPE__)d;
}
