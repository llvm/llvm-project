#import <Foundation/Foundation.h>

@interface IAmBlocky : NSObject
{
  @public
  int blocky_ivar;
}
- (IAmBlocky *) init;
- (int) callABlock: (int) block_value;
@end
