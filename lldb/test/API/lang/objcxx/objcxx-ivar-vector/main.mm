#import <Foundation/Foundation.h>

#include <vector>

@interface MyElement : NSObject {
}
@end

@interface MyClass : NSObject {
  std::vector<MyElement *> elements;
};

-(void)doSomething;

@end

@implementation MyClass

-(void)doSomething
{
  NSLog(@"Hello"); //% self.expect("expression -- elements", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["vector", "MyElement"]);
}

@end

int main ()
{
  @autoreleasepool
  {
    MyClass *c = [MyClass alloc];
    [c doSomething];
  }
}
