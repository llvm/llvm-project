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

@interface Calc : NSObject
- (int)addFortyTwoTo:(int)value;
@end
@implementation Calc
- (int)addFortyTwoTo:(int)value {
  return value + 42;
}
@end

// Message sends in expressions require the JIT'd module's selectors to be
// registered with the runtime (the GNUstep plugin's IR pass does this);
// without it the dispatch reaches the runtime with an unregistered selector.
//
// RUN: %lldb -b -o "b objc-gnustep-expr.m:46" -o "run" \
// RUN:          -o "expr [c addFortyTwoTo:100]" \
// RUN:          -o "expr (int)[[Calc new] addFortyTwoTo:1]" -- %t | FileCheck %s
//
int main() {
  Calc *c = [Calc new];
  (void)[c addFortyTwoTo:0];
  return 0;
}
//
// CHECK: (lldb) expr [c addFortyTwoTo:100]
// CHECK: (int) {{\$[0-9]+}} = 142
//
// CHECK: addFortyTwoTo:1]
// CHECK: (int) {{\$[0-9]+}} = 43
