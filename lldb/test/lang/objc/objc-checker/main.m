#import <Foundation/Foundation.h>

@interface Simple : NSObject
{
  int _value;
}
- (int) value;
- (void) setValue: (int) newValue;
@end

@implementation Simple
- (int) value
{
  return _value;
}

- (void) setValue: (int) newValue
{
  _value = newValue;
}
@end

int main ()
{
  NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];
  Simple *my_simple = [[Simple alloc] init];
  my_simple.value = 20;
  // Set a breakpoint here.
  NSLog (@"Object has value: %d.", my_simple.value); 
  [pool drain];
  return 0;
}
