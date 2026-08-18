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

@interface Doubler : NSObject
- (int)twice:(int)value;
@end
@implementation Doubler
- (int)twice:(int)value {
  return value * 2;
}
@end

// Stepping at a message send has to run through the runtime's dispatch
// function and land in the method implementation.
//
// RUN: %lldb -b -o "b objc-gnustep-stepping.m:53" -o "run" -o "step" \
// RUN:     -- %t | FileCheck %s --check-prefix=STEP_IN
//
// A message to nil dispatches nowhere, so the step must not try to run to an
// implementation - and must not strand the user in the runtime's dispatch
// assembly either. A runtime built from source has line information for that
// assembly, so without stepping back out the debugger would stop in
// objc_msgSend.S; asserting on the frame's source location is what catches
// that, since the function name alone looks plausible either way.
//
// RUN: %lldb -b -o "b objc-gnustep-stepping.m:55" -o "run" -o "step" \
// RUN:     -o "frame info" -- %t | FileCheck %s --check-prefix=STEP_OVER_NIL
//
int main() {
  Doubler *doubler = [Doubler new];
  int value = [doubler twice:21]; // line 53: STEP_IN breaks here
  Doubler *nothing = (Doubler *)0;
  int none = [nothing twice:1]; // line 55: STEP_OVER_NIL breaks here
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
