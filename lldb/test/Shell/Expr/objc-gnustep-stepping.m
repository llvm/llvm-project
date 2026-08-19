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

@interface Doubler : NSObject
- (int)twice:(int)value;
@end
@implementation Doubler
- (int)twice:(int)value {
  return value * 2;
}
@end

//
// RUN: %lldb %inferior_abi -b -o "b objc-gnustep-stepping.m:49" -o "run" -o "step" \
// RUN:     -- %t | FileCheck %s --check-prefix=STEP_IN
//
// A message to nil dispatches nowhere, so the step must land back in the
// caller rather than in the runtime's dispatch assembly. The assertion is on
// the frame's source location because the function name alone looks
// plausible either way.
//
// RUN: %lldb %inferior_abi -b -o "b objc-gnustep-stepping.m:51" -o "run" -o "step" \
// RUN:     -o "frame info" -- %t | FileCheck %s --check-prefix=STEP_OVER_NIL
//
int main() {
  Doubler *doubler = [Doubler new];
  int value = [doubler twice:21]; // STEP_IN breaks here
  Doubler *nothing = (Doubler *)0;
  int none = [nothing twice:1]; // STEP_OVER_NIL breaks here
  return value + none;
}
//
// STEP_IN: (lldb) step
// STEP_IN: stop reason = step in
// STEP_IN: -[Doubler twice:]
//
// STEP_OVER_NIL: (lldb) step
// STEP_OVER_NIL: stop reason = step in
// STEP_OVER_NIL-NOT: -[Doubler twice:]
// STEP_OVER_NIL: (lldb) frame info
// STEP_OVER_NIL: frame #0: {{.*}}main at objc-gnustep-stepping.m:{{[0-9]+}}
