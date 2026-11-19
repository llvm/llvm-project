// UNSUPPORTED: system-linux, system-windows

// RUN: %clang_host -g %s -o %t -framework Foundation
// RUN: %lldb %t \
// RUN: -o "br set -p 'I am about to throw'" \
// RUN: -o "run" \
// RUN: -o "expr -e false -- (int)[my_class iCatchMyself]" \
// RUN: 2>&1 | FileCheck %s --check-prefix=NO-TRAP

// NO-TRAP: (int) ${{[0-9]+}} = 57

// RUN: %lldb %t \
// RUN: -o "br set -p 'I am about to throw'" \
// RUN: -o "run" \
// RUN: -o "expr -e true -- (int)[my_class iCatchMyself]" \
// RUN: 2>&1 | FileCheck %s --check-prefix=TRAP

// TRAP: error: Expression execution was interrupted: internal ObjC exception breakpoint

#import <Foundation/Foundation.h>

@interface MyClass : NSObject {
}
- (int)callMeIThrow;
- (int)iCatchMyself;
@end

@implementation MyClass
- (int)callMeIThrow {
  NSException *e = [NSException exceptionWithName:@"JustForTheHeckOfItException" reason:@"I felt like it" userInfo:nil];
  @throw e;
  return 56;
}
- (int)iCatchMyself {
  int return_value = 55;
  @try {
    return_value = [self callMeIThrow];
  } @catch (NSException *e) {
    return_value = 57;
  }

  return return_value;
}
@end

int main() {
  int return_value;
  MyClass *my_class = [[MyClass alloc] init];
  NSLog(@"I am about to throw.");

  return_value = [my_class iCatchMyself];

  return return_value;
}
