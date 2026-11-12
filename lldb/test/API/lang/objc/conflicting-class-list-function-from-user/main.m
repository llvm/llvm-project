#import <Foundation/Foundation.h>
#include <stdlib.h>

// A function with this signature will be called by LLDB to retrieve the
// Objective-C class list. We shouldn't call this function that is defined
// by the user if possible.
Class *objc_copyRealizedClassList_nolock(unsigned int *outCount) {
  // Don't try to implement this properly but just abort.
  abort();
}

// Define some custom class that makes LLDB read the Objective-C class list.
@interface CustomClass : NSObject {
};
@end
@implementation CustomClass
@end

int main(int argc, char **argv) {
  id custom_class = [[CustomClass alloc] init];
  // Make sure our trap function is emitted but never called (the test doesn't
  // launch the executable with any args).
  if (argc == 123) {
    objc_copyRealizedClassList_nolock(0);
  }
  return 0; // break here
}
