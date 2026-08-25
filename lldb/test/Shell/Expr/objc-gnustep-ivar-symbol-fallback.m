// REQUIRES: objc-gnustep
//
// RUN: %build %inferior_target %s --compiler=clang --objc-gnustep --output=%t
//
// An expression that reads an ivar refers to the runtime's offset variable by
// symbol. Normally the inferior exports it; remove it to reach the path that
// has to recover the address from the runtime's own metadata instead. That is
// what a stripped module looks like to an expression.
//
// RUN: llvm-objcopy --strip-symbol=__objc_ivar_offset_Holder.value.i \
// RUN:              --strip-symbol=__objc_ivar_offset_Holder.tail.i %t %t.stripped
// RUN: llvm-nm %t.stripped | FileCheck --check-prefix=NOSYM %s
//
// NOSYM-NOT: __objc_ivar_offset_Holder
//
// RUN: %lldb %inferior_abi -b -o "breakpoint set -p \"break [h]ere\" -X main" -o "run" \
// RUN:          -o "expr holder->value" -o "expr holder->tail" -- %t.stripped \
// RUN:     | FileCheck %s
//
// The checks live above the code deliberately: lldb echoes the source lines
// around the breakpoint, so a CHECK sitting within three lines of it matches
// its own text in that echo rather than the command output.
//
// CHECK: (lldb) expr holder->value
// CHECK: (int) $0 = 11
// CHECK: (lldb) expr holder->tail
// CHECK: (int) $1 = 22

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

@interface Holder : NSObject {
@public
  int value;
  int tail;
}
@end
@implementation Holder
@end

int main() {
  Holder *holder = (Holder *)[Holder new];
  holder->value = 11;
  holder->tail = 22;
  return 0; // break here
}
