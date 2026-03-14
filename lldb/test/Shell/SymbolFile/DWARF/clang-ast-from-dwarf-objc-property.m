// REQUIRES: system-darwin
//
// RUN: %clang_host -g -c -o %t.o %s
// RUN: lldb-test symbols -dump-clang-ast %t.o | FileCheck %s

__attribute__((objc_root_class))
@interface Root
@property (readonly) int ro_number;
@property int rw_number;
@property (readonly, getter=custom_getter) int manual;
- (int)custom_getter;
@property (class, readonly) int class_property;
@end

Root *obj;

// CHECK: |-ObjCPropertyDecl {{.*}} ro_number 'int' readonly
// CHECK: | `-getter ObjCMethod [[READONLY:0x[0-9a-f]+]] 'ro_number'
// CHECK: |-ObjCMethodDecl [[READONLY]] {{.*}} implicit - ro_number 'int'
// CHECK: |-ObjCPropertyDecl {{.*}} rw_number 'int' assign readwrite
// CHECK: | |-getter ObjCMethod {{.*}} 'rw_number'
// CHECK: | `-setter ObjCMethod {{.*}} 'setRw_number:'
// CHECK: |-ObjCPropertyDecl {{.*}} manual 'int' readonly
// CHECK: | `-getter ObjCMethod [[CUSTOM:0x[0-9a-f]+]] 'custom_getter'
// CHECK: |-ObjCMethodDecl [[CUSTOM]] {{.*}} - custom_getter 'int'
// CHECK: |-ObjCPropertyDecl {{.*}} class_property 'int' readonly class
// CHECK: | `-getter ObjCMethod [[CLASS:0x[0-9a-f]+]] 'class_property'
// CHECK: `-ObjCMethodDecl [[CLASS]] {{.*}} + class_property 'int'

