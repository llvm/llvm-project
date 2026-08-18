// REQUIRES: objc-gnustep
//
// RUN: %build %inferior_target %s --compiler=clang --objc-gnustep --output=%t

#import "objc/runtime.h"

#ifdef __has_attribute
#if __has_attribute(objc_root_class)
__attribute__((objc_root_class))
#endif
#endif
@interface NSObject {
  id isa;
  int refcount;
}
+ (id)new;
@end
@implementation NSObject
+ (id)new {
  return class_createInstance(self, 0);
}
@end

@interface Compiled : NSObject
@end
@implementation Compiled
@end

// RUN: %lldb %inferior_abi -b -o "breakpoint set -p \"break [h]ere\" -X main" -o "run" \
// RUN:          -o "frame variable -d run-target made compiled" -- %t \
// RUN:     | FileCheck %s
//
int main() {
  // A class registered at run time has no class symbol and no debug info, so
  // its name and layout are only ever available from the runtime's metadata.
  Class pair = objc_allocateClassPair(objc_getClass("NSObject"), "MadeAtRuntime", 0);
  objc_registerClassPair(pair);
  id made = class_createInstance(pair, 0);
  id compiled = [Compiled new];
  return made == compiled; // break here
}
//
// Both are declared `id`, so each has to be resolved through the runtime. The
// compiled class also has debug info; the runtime-created one has nothing but
// the metadata, and must still report its own name rather than falling back to
// the declared type.
//
// CHECK: (lldb) frame variable -d run-target made compiled
// CHECK-NEXT: (MadeAtRuntime *) made
// CHECK-NEXT: (Compiled *) compiled
