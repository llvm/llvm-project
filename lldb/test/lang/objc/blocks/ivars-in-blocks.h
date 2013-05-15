#import <Foundation/Foundation.h>

@interface IAmBlocky : NSObject
{
  @public
  int blocky_ivar;
}
+ (void) classMethod;
- (IAmBlocky *) init;
- (int) callABlock: (int) block_value;
@end
