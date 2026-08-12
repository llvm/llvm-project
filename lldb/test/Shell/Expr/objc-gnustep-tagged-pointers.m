// REQUIRES: objc-gnustep
//
// RUN: %build %s --compiler=clang --objc-gnustep --output=%t

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

// libobjc2 calls these "small objects": a pointer with any of its low bits set
// holds a value inline rather than pointing at an object, and its class comes
// from the runtime's table of registered small object classes.
@interface TinyNumber : NSObject
@end
@implementation TinyNumber
@end

@interface Ordinary : NSObject
@end
@implementation Ordinary
@end

// RUN: %lldb -b -o "b objc-gnustep-tagged-pointers.m:49" -o "run" \
// RUN:          -o "frame variable -d run-target tagged" \
// RUN:          -o "frame variable -d run-target ordinary" -- %t | FileCheck %s
//
int main() {
  objc_registerSmallObjectClass_np(objc_getClass("TinyNumber"), 1);

  // A payload of 3 in the slot registered above.
  id tagged = (id)(uintptr_t)((3 << 3) | 1);
  id ordinary = [Ordinary new];
  return tagged == ordinary;
}
//
// The tagged value's class comes from the runtime's small object table, while
// an ordinary pointer is still resolved by reading its class structure.
//
// CHECK: (lldb) frame variable -d run-target tagged
// CHECK: (TinyNumber *) tagged = 0x{{0*}}19
//
// CHECK: (lldb) frame variable -d run-target ordinary
// CHECK: (Ordinary *) ordinary = 0x
