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

// libobjc2 aligns each ivar with the hidden reference-count word included, so
// an ivar needing more than pointer alignment does not land where laying the
// class out as a plain struct would put it. __int128 rather than long double
// because it is over-aligned on every target this runs on: long double is
// 8-byte aligned for MSVC, which would make this vacuous there.
@interface Wide : NSObject {
@public
  int before;
  __int128 aligned;
  int after;
}
@end
@implementation Wide
@end

// RUN: %lldb %inferior_abi -b -o "breakpoint set -p \"break [h]ere\" -X main" -o "run" \
// RUN:          -o "frame variable *wide" -o "p wide->aligned" -- %t \
// RUN:     | FileCheck %s
//
int main() {
  Wide *wide = (Wide *)[Wide new];
  wide->before = 1;
  wide->aligned = 42;
  wide->after = 3;
  return 0; // break here
}
//
// The aggregate must agree with direct access, for the over-aligned ivar and
// for every ivar after it.
//
// CHECK: (lldb) frame variable *wide
// CHECK:   before = 1
// CHECK-NEXT:   aligned = 42
// CHECK-NEXT:   after = 3
//
// CHECK: (lldb) p wide->aligned
// CHECK: (__int128) 42
