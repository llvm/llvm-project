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
// RUN: %lldb %inferior_abi -b -o "b objc-gnustep-expr.m:56" -o "run" \
// RUN:          -o "expr [c addFortyTwoTo:100]" \
// RUN:          -o "expr (int)[[Calc new] addFortyTwoTo:1]" -- %t | FileCheck %s
//
// The checks sit above the code: lldb echoes the source lines around the
// breakpoint, so a CHECK within three lines of it matches its own text in
// that echo rather than the command output.
//
// CHECK: (lldb) expr [c addFortyTwoTo:100]
// CHECK: (int) {{\$[0-9]+}} = 142
//
// CHECK: addFortyTwoTo:1]
// CHECK: (int) {{\$[0-9]+}} = 43
//
int main() {
  Calc *c = [Calc new];
  (void)[c addFortyTwoTo:0];
  return 0;
}
