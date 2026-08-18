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

// gnustep-base spells the wide flag as a one-bit field of an anonymous struct
// (`struct { unsigned wide:1; ... } _flags`, Source/GSPrivate.h), which
// libobjc2's type encodings cannot describe - so where the class has no debug
// info the summary provider has no `wide` member to read and falls back to
// reading the bit out of memory. Declaring it as a plain word here leaves the
// member missing in the same way, which is what takes that path.
//
// The name is what matters: formatters are matched on the class name the
// runtime reports, so this stands in for gnustep-base's own GSCInlineString.
@interface GSCInlineString : NSObject {
@public
  char *_contents;
  unsigned _count;
  unsigned _flags;
}
@end
@implementation GSCInlineString
@end

int main() {
  GSCInlineString *narrow = [GSCInlineString new];
  narrow->_contents = "hi";
  narrow->_count = 2;
  narrow->_flags = 0;

  // Bit 0 is `wide`. Set it and the same bytes are read as UTF-16, which is
  // what proves the fallback read the flag rather than defaulting to narrow.
  GSCInlineString *wide = [GSCInlineString new];
  wide->_contents = "h\0i\0";
  wide->_count = 2;
  wide->_flags = 1;

  return 0; // break here
}

// RUN: %lldb %inferior_abi -b -o "b objc-gnustep-string-flags.m:56" -o "run" \
// RUN:     -o "frame variable -d run-target narrow" \
// RUN:     -o "frame variable -d run-target wide" \
// RUN:     -- %t | FileCheck %s
//
// CHECK: (lldb) frame variable -d run-target narrow
// CHECK: @"hi"
//
// CHECK: (lldb) frame variable -d run-target wide
// CHECK: @"hi"
