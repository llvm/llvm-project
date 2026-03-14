#import <Test/Test.h>
#import <TestExt/TestExt.h>

int main() {
  @autoreleasepool {
    Test *test = [[Test alloc] init];
    [test doSomethingElse:&test->_range];
  }
}    

