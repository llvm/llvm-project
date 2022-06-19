#import <Foundation/Foundation.h>

// Observable side effect that is changed when one of our trap functions is
// called. This should always retain its initial value in a successful test run.
const char *called_function = "none";

// Below several trap functions are declared in different scopes that should
// never be called even though they share the name of some of the utility
// functions that LLDB has to call when updating the Objective-C class list
// (i.e. 'free' and 'objc_copyRealizedClassList_nolock').
// All functions just indicate that they got called by setting 'called_function'
// to their own name.

namespace N {
void free(void *) { called_function = "N::free"; }
void objc_copyRealizedClassList_nolock(unsigned int *) {
  called_function = "N::objc_copyRealizedClassList_nolock";
}
}

struct Context {
  void free(void *) { called_function = "Context::free"; }
  void objc_copyRealizedClassList_nolock(unsigned int *) {
    called_function = "Context::objc_copyRealizedClassList_nolock";
  }
};

@interface ObjCContext : NSObject {
}
- (void)free:(void *)p;
- (void)objc_copyRealizedClassList_nolock:(unsigned int *)outCount;
@end

@implementation ObjCContext
- (void)free:(void *)p {
  called_function = "ObjCContext::free";
}

- (void)objc_copyRealizedClassList_nolock:(unsigned int *)outCount {
  called_function = "ObjCContext::objc_copyRealizedClassList_nolock";
}
@end

int main(int argc, char **argv) {
  id str = @"str";
  // Make sure all our conflicting functions/methods are emitted. The condition
  // is never executed in the test as the process is launched without args.
  if (argc == 1234) {
    Context o;
    o.free(nullptr);
    o.objc_copyRealizedClassList_nolock(nullptr);
    N::free(nullptr);
    N::objc_copyRealizedClassList_nolock(nullptr);
    ObjCContext *obj = [[ObjCContext alloc] init];
    [obj free:nullptr];
    [obj objc_copyRealizedClassList_nolock:nullptr];
  }
  return 0; // break here
}
