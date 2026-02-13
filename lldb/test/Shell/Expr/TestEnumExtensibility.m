// UNSUPPORTED: system-linux, system-windows

// RUN: %clangxx_host %s -c -g -o %t
// RUN: %lldb %t \
// RUN:   -o "target var gClosed gOpen gNS gNSOpts" \
// RUN:   -o "image dump ast" \
// RUN:   2>&1 | FileCheck %s

#import <Foundation/Foundation.h>

enum __attribute__((enum_extensibility(closed))) Closed { C1 } gClosed;

enum __attribute__((enum_extensibility(open))) Open { O1 } gOpen;

typedef NS_ENUM(int, NS) { N1 } gNS;

typedef NS_OPTIONS(int, NSO) { OPT1 } gNSOpts;

// CHECK:      EnumDecl {{.*}} Closed
// CHECK-NEXT: |-EnumExtensibilityAttr {{.*}} Closed
// CHECK-NEXT: `-EnumConstantDecl {{.*}} C1 'Closed'

// CHECK:      EnumDecl {{.*}} Open
// CHECK-NEXT: |-EnumExtensibilityAttr {{.*}} Open
// CHECK-NEXT: `-EnumConstantDecl {{.*}} O1 'Open'

// CHECK:      EnumDecl {{.*}} NS
// CHECK-NEXT: |-EnumExtensibilityAttr {{.*}} Open
// CHECK-NEXT: `-EnumConstantDecl {{.*}} N1 'NS'

// CHECK:      EnumDecl {{.*}} NSO
// CHECK-NEXT: |-EnumExtensibilityAttr {{.*}} Open
// CHECK-NEXT: `-EnumConstantDecl {{.*}} OPT1 'NSO'
