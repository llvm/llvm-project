#import <Foundation/Foundation.h>

@interface MyClass : NSObject {
@public
  int _foo;
};

-(id)init;
@end

@implementation MyClass

-(id)init
{
  if ([super init])
  {
    _foo = 3;
  }

  return self;
}

@end

int main ()
{
  @autoreleasepool
  {
    MyClass *mc = [[MyClass alloc] init];

    NSLog(@"%d", mc->_foo); // Set breakpoint here.
  }
}
