// REQUIRES: objc-gnustep
//
// RUN: %build %s --compiler=clang --objc-gnustep --output=%t

#import "objc/runtime.h"

@protocol NSCoding
@end

// A root class may declare its `isa` as `id` rather than `Class` (the GNUstep
// tests and many libobjc2 programs do). Because `id` can carry a dynamic
// type, the value of such a field - a class object - is offered to the
// runtime for dynamic typing. libobjc2 names a metaclass after its class, so
// a naive runtime would then report the class object as an instance of the
// class, and expanding it would recurse forever through the same `isa`.
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

// (No ivars beyond the root class's: clang trips an assertion compiling
// some GNUstep classes with ivars in +assertions builds, see
// objc-gnustep-print.m.)
@interface Base : NSObject
@end
@implementation Base
@end

@interface Derived : Base
@end
@implementation Derived
@end

// RUN: %lldb -b -o "b objc-gnustep-class-objects.m:53" -o "run" \
// RUN:          -o "frame variable -d run-target -T object" \
// RUN:          -o "frame variable -d run-target -T *object" \
// RUN:          -o "frame variable -d run-target -T object->isa" \
// RUN:          -- %t | FileCheck %s
//
int main() {
  Base *object = [Derived new];
  (void)object;
  return object == 0;
}
//
// The object itself gets its dynamic type...
// CHECK: (lldb) frame variable -d run-target -T object
// CHECK: (Derived *) object = 0x
//
// ...and its `isa` stays a plain `id`: it points at the class object, which
// must not be presented as an instance.
// CHECK: (lldb) frame variable -d run-target -T *object
// CHECK: (Derived) *object = {
// CHECK: (id) isa = 0x
// CHECK-NOT: (Derived *) isa
// CHECK-NOT: (Base *) isa
// CHECK: (int) refcount
//
// CHECK: (lldb) frame variable -d run-target -T object->isa
// CHECK: (id) object->isa = 0x
