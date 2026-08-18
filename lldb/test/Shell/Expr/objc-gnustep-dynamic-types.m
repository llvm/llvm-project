// REQUIRES: objc-gnustep
//
// RUN: %build %inferior_target %s --compiler=clang --objc-gnustep --output=%t

#import "objc/runtime.h"

@protocol NSCoding
@end

#ifdef __has_attribute
#if __has_attribute(objc_root_class)
__attribute__((objc_root_class))
#endif
#endif
@interface NSObject <NSCoding> {
  id isa;
  int refcount;
}
@end
@implementation NSObject
+ (id)new {
  return class_createInstance(self, 0);
}
@end

@interface Base : NSObject
@end
@implementation Base
@end

@interface Derived : Base
@end
@implementation Derived
@end

// The static type of `object` is Base, but the dynamic type is Derived. The
// GNUstep runtime resolves the dynamic type by reading the class structure
// from the inferior's memory and attaching the matching type from debug info.
//
// RUN: %lldb %inferior_abi -b -o "b objc-gnustep-dynamic-types.m:46" -o "run" \
// RUN:          -o "frame variable -d run-target object" \
// RUN:          -o "frame variable -d no-dynamic-values object" -- %t | FileCheck %s
//
int main() {
  Base *object = [Derived new];
  (void)object;
  return 0;
}
//
// CHECK: (lldb) frame variable -d run-target object
// CHECK: (Derived *) object = 0x
//
// CHECK: (lldb) frame variable -d no-dynamic-values object
// CHECK: (Base *) object = 0x
