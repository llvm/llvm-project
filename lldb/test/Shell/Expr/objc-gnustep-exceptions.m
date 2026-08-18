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
}
+ (id)new;
@end
@implementation NSObject
+ (id)new {
  return class_createInstance(self, 0);
}
@end

// `po` on the exception object goes through -description/-UTF8String, so a
// minimal pair of those is all this needs; nothing here uses Foundation.
@interface Str : NSObject {
  const char *_bytes;
}
+ (id)withBytes:(const char *)bytes;
- (const char *)UTF8String;
@end
@implementation Str
+ (id)withBytes:(const char *)bytes {
  Str *str = [Str new];
  str->_bytes = bytes;
  return str;
}
- (const char *)UTF8String {
  return _bytes;
}
@end

@interface Boom : NSObject
- (id)description;
@end
@implementation Boom
- (id)description {
  return [Str withBytes:"<Boom: thrown>"];
}
@end

void thrower(void) { @throw [Boom new]; }

const char *g_caught_name = 0;

int main() {
  @try {
    thrower();
  } @catch (id caught) {
    // Something with real code, so a breakpoint can land inside the handler.
    g_caught_name = object_getClassName(caught);
  }
  return g_caught_name == 0;
}

// An Objective-C exception breakpoint stops where the exception is raised,
// and the frame recognizer presents the thrown object as `exception`.
//
// RUN: %lldb -b -o "breakpoint set -E objc" -o "run" -o "frame variable" \
// RUN:     -o "thread exception" \
// RUN:     -- %t | FileCheck %s --check-prefix=THROW
//
// THROW: (lldb) breakpoint set -E objc
// THROW: Breakpoint {{[0-9]+}}:
//
// THROW: (lldb) run
// THROW: stop reason = hit Objective-C exception
//
// The recognizer synthesizes the argument, so the thrown object shows up in
// `frame variable` even though objc_exception_throw has no debug info - and
// it carries the dynamic type, not the `id` the runtime declares.
// THROW: (lldb) frame variable
// THROW: (Boom *) exception = 0x
//
// THROW: (lldb) thread exception
// THROW: (Boom *) exception = 0x

// The object is reachable as a real local inside the handler, where `po`
// works on it normally.
//
// RUN: %lldb -b -o "b objc-gnustep-exceptions.m:63" -o "run" -o "po caught" \
// RUN:     -- %t | FileCheck %s --check-prefix=CAUGHT
//
// CAUGHT: (lldb) po caught
// CAUGHT: <Boom: thrown>
// CAUGHT-NOT: warning: `po` was unsuccessful
