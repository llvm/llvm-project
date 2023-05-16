// REQUIRES: objc-gnustep
// XFAIL: system-windows
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
- (id)class {
  return object_getClass(self);
}
+ (id)new {
  return class_createInstance(self, 0);
}
@end

@interface TestObj : NSObject {}
- (int)ok;
@end
@implementation TestObj
- (int)ok {
  return self ? 0 : 1;
}
@end

// RUN: %lldb -b -o "b objc-gnustep-print.m:35" -o "run" -o "p self" -o "p *self" -- %t | FileCheck %s --check-prefix=SELF
//
// SELF: (lldb) b objc-gnustep-print.m:35
// SELF: Breakpoint {{.*}} at objc-gnustep-print.m
//
// SELF: (lldb) run
// SELF: Process {{[0-9]+}} stopped
// SELF: -[TestObj ok](self=[[SELF_PTR:0x[0-9a-f]+]]{{.*}}) at objc-gnustep-print.m:35
//
// SELF: (lldb) p self
// SELF: (TestObj *) [[SELF_PTR]]
//
// SELF: (lldb) p *self
// SELF: (TestObj) {
// SELF:   NSObject = {
// SELF:     isa
// SELF:     refcount
// SELF:   }
// SELF: }

int main() {
  TestObj *t = [TestObj new];
  return [t ok];
}
