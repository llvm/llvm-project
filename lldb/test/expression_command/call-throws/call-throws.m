#import <Foundation/Foundation.h>

@interface MyClass : NSObject
{
}
- (int) callMeIThrow;
@end

@implementation MyClass
- (int) callMeIThrow
{
    NSException *e = [NSException
                       exceptionWithName:@"JustForTheHeckOfItException"
                       reason:@"I felt like it"
                       userInfo:nil];
    @throw e;
    return 56;
}
@end

int
main ()
{
  int return_value;
  MyClass *my_class = [[MyClass alloc] init];

  NSLog (@"I am about to throw.");

  return_value = [my_class callMeIThrow];

  return return_value;
}
