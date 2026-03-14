#import <objc/NSObject.h>
#import "my-base.h"
@implementation MyBase
#if __OBJC2__
@synthesize propertyMovesThings;
#else
@synthesize propertyMovesThings = maybe_used;
#endif
@end

