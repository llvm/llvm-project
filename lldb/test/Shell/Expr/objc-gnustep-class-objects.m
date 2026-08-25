// REQUIRES: objc-gnustep
//
// RUN: %build %inferior_target %s --compiler=clang --objc-gnustep --output=%t

#import "objc/runtime.h"

@protocol NSCoding
@end

// A root class may declare its `isa` as `id` rather than `Class`, and `id`
// carries a dynamic type - so a class object gets offered to the runtime for
// typing. libobjc2 names a metaclass after its class, which is what makes
// reporting a class object as an instance of that class possible.
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

// RUN: %lldb %inferior_abi -b -o "b objc-gnustep-class-objects.m:68" -o "run" \
// RUN:          -o "frame variable -d run-target -T object" \
// RUN:          -o "frame variable -d run-target -T *object" \
// RUN:          -o "frame variable -d run-target -T object->isa" \
// RUN:          -- %t | FileCheck %s
//
// The checks sit above the code: lldb echoes the source lines around the
// breakpoint, so a CHECK within three lines of it matches its own text in
// that echo rather than the command output.
//
// CHECK: (lldb) frame variable -d run-target -T object
// CHECK: (Derived *) object = 0x
//
// The `isa` must stay a plain `id`: it points at the class object, which is
// not an instance of the class.
// CHECK: (lldb) frame variable -d run-target -T *object
// CHECK: (Derived) *object = {
// CHECK: (id) isa = 0x
// CHECK-NOT: (Derived *) isa
// CHECK-NOT: (Base *) isa
// CHECK: (int) refcount
//
// CHECK: (lldb) frame variable -d run-target -T object->isa
// CHECK: (id) object->isa = 0x
//
int main() {
  Base *object = [Derived new];
  (void)object;
  return object == 0;
}
