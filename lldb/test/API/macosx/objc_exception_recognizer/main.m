#import <Foundation/Foundation.h>

@interface MyException : NSException
{
  int extra_info;
}
- (NSException *) initWithExtraInfo: (int) info;
@end

@implementation MyException
- (NSException *) initWithExtraInfo: (int) info
{
  [super initWithName: @"NSException" reason: @"Simple Reason" userInfo: nil];
  self->extra_info = info;
  return self;
}
@end

int
main(int argc, char **argv)
{
  // Set a breakpoint here for plain exception:
  @try {
    NSException *plain_exc = [[NSException alloc] initWithName: @"NSException" reason: @"Simple Reason" userInfo: nil];
    [plain_exc raise];
  }
  @catch (id anException) {}

  // Set a breakpoint here for MyException:
  @try {
    MyException *my_exc = [[MyException alloc] initWithExtraInfo: 100];
    [my_exc raise];
  }
  @catch (id anException) {}

  return 0;
}
