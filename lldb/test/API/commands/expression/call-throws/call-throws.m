#import <Foundation/Foundation.h>

@interface MyClass : NSObject
{
}
- (int) callMeIThrow;
- (int) iCatchMyself;
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

- (int) iCatchMyself
{
  int return_value = 55;
  @try
    {
      return_value = [self callMeIThrow];
    }
  @catch (NSException *e)
    {
      return_value = 57;
    }
  return return_value;
}
@end

int
main ()
{
  int return_value;
  MyClass *my_class = [[MyClass alloc] init];

  NSLog (@"I am about to throw.");

  return_value = [my_class iCatchMyself];

  return return_value;
}
