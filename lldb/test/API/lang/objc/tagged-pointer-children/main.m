#import <Foundation/Foundation.h>

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    // A single-index NSIndexSet is stored as an Objective-C tagged pointer.
    // It has a summary but no ivars or base classes LLDB could materialize
    // from memory.
    NSIndexSet *tagged = [NSIndexSet indexSetWithIndex:1];

    // A discontiguous set cannot be tagged, so it is a real heap object with a
    // readable isa and ivar layout.
    NSMutableIndexSet *heap = [NSMutableIndexSet indexSet];
    [heap addIndex:1];
    [heap addIndex:1000000];

    NSLog(@"%@ %@", tagged, heap); // break here
  }
  return 0;
}
