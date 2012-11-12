#import "objc-ivar-offsets.h"

@implementation BaseClass
@synthesize backed_int = _backed_int;
#if __OBJC2__
@synthesize unbacked_int;
#else
@synthesize unbacked_int = _unbacked_int;
#endif
@end

@implementation DerivedClass
@synthesize derived_backed_int = _derived_backed_int;
#if __OBJC2__
@synthesize derived_unbacked_int;
#else
@synthesize derived_unbacked_int = _derived_unbacked_int;
#endif
@end
