#include <Foundation/Foundation.h>
#include <dlfcn.h>
#include <libgen.h>

@protocol FooProtocol
+ (void)f;
@end

int main(int argc, const char **argv) {
  char dylib[4096];
  strlcpy(dylib, dirname(argv[0]), 4096);
  strlcat(dylib, "/libNewerTarget.dylib", 4096);
  dlopen(dylib, RTLD_LAZY);
  Class<FooProtocol> fooClass = NSClassFromString(@"NewerTarget.Foo");
  [[[fooClass alloc] init] f];
  return 0;
}

