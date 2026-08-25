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
+ (id)new;
@end
@implementation NSObject
+ (id)new {
  return class_createInstance(self, 0);
}
@end

@interface Widget : NSObject {
@public
  int count;
}
- (int)twice:(int)n;
@end
@implementation Widget
- (int)twice:(int)n { return n * 2; }
@end

struct CppThing {
  int a;
};

// A message send in an Objective-C++ frame needs the same selector
// registration as one in Objective-C: the expression's selectors are only
// interned by the runtime if the IR pass runs, and the runtime is looked up
// by the expression's language.
//
// RUN: %lldb %inferior_abi -b -o "breakpoint set -p \"break [h]ere\" -X main" \
// RUN:          -o "run" -o "expr (int)[w twice:21]" \
// RUN:          -o "frame variable *w" -o "frame variable t" -- %t | FileCheck %s
//
// The checks sit above the code deliberately: lldb echoes the source lines
// around the breakpoint, so a CHECK within three lines of it would match its
// own text in that echo rather than the command output.
//
// CHECK: (lldb) expr (int)[w twice:21]
// CHECK: (int) $0 = 42
//
// CHECK: (lldb) frame variable *w
// CHECK:   count = 7
//
// CHECK: (lldb) frame variable t
// CHECK: (CppThing) t = (a = 3)
//
int main() {
  Widget *w = (Widget *)[Widget new];
  w->count = 7;
  struct CppThing t = {3};
  return t.a; // break here
}
