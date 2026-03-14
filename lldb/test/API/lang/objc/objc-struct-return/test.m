#import <Foundation/Foundation.h>

@interface RangeProvider : NSObject {
};
-(NSRange)getRange;
@end

@implementation RangeProvider
-(NSRange)getRange
{
  return NSMakeRange(0, 3);
}
@end

int main()
{
  @autoreleasepool
  {
    RangeProvider *provider = [RangeProvider alloc];
    NSRange range = [provider getRange]; // Set breakpoint here.
    return 0;
  }
}
