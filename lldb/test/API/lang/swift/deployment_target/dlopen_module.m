#include <Foundation/Foundation.h>
#include <dlfcn.h>
#include <libgen.h>
#include <limits.h>

@protocol FooProtocol
- (void)f;
@end

int main(int argc, const char **argv) {
  char dylib[PATH_MAX];
  strlcpy(dylib, dirname(argv[0]), PATH_MAX);
  strlcat(dylib, "/libNewerTarget.dylib", PATH_MAX);
  dlopen(dylib, RTLD_LAZY);
  Class<FooProtocol> fooClass = NSClassFromString(@"NewerTarget.Foo");
  [[[fooClass alloc] init] f];
  return 0;
}
