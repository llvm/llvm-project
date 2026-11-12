#import <Foundation/Foundation.h>

void baz() {}

struct MyClass {
  void bar() {
    baz(); // break here
  }
};

@interface MyObject : NSObject {}
- (void)foo;
@end

@implementation MyObject
- (void)foo {
  MyClass c;
  c.bar(); // break here
}
@end

int main (int argc, char const *argv[]) {
    NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];
    id obj = [MyObject new];
    [obj foo];
    [pool release];
    return 0;
}
