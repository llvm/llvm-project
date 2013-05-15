#import "Bar.h"

@interface InternalClass : NSObject {
    @public
    NSString *foo;
    NSString *bar;
}
@end

@implementation InternalClass
@end

@interface Bar () 
{
    NSString *_hidden_ivar;
}

@end

@implementation Bar

- (id)init
{
    self = [super init];
    if (self) {
        _hidden_ivar = [NSString stringWithFormat:@"%p: @Bar", self];
    }
    return self; // Set breakpoint where Bar is an implementation
}

- (void)dealloc
{
    [_hidden_ivar release];
    [super dealloc];
}

- (NSString *)description
{
    return [_hidden_ivar copyWithZone:NULL];
}

@end
 