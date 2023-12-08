#import <objc/NSObject.h>

@interface Foo : NSObject {
}
-(int)get;
@end

@implementation Foo
-(int)get
{
  return 1;
}
@end

@interface Bar : Foo {
}
-(int)get;
@end

@implementation Bar
-(int)get
{
  return 2; 
}

-(int)callme
{
  return [self get]; // Set breakpoint here.
}
@end

int main()
{
  @autoreleasepool
  {
    Bar *bar = [Bar alloc];
    return [bar callme];
  }
}
