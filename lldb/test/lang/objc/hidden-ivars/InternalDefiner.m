#import "InternalDefiner.h"

@interface InternalDefiner () {
    uintptr_t bar;
}

@end

@implementation InternalDefiner

-(id)init
{
    if (self = [super init])
    {
        foo = 2;
        bar = 3;
    }
    return self;
}

-(id)initWithFoo:(uintptr_t)f andBar:(uintptr_t)b
{
    if (self = [super init])
    {
        foo = f;
        bar = b;
    }
    return self;
}

@end
