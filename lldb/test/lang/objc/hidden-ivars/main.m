#import <Foundation/Foundation.h>
#import "InternalDefiner.h"

@interface Container : NSObject {
@public
    InternalDefiner *_definer;
}

-(id)init;
@end

@implementation Container

-(id)init
{
    if (self = [super init])
    {
        _definer = [[InternalDefiner alloc] initWithFoo:4 andBar:5];
    }
    return self;
}

@end

@interface InheritContainer : InternalDefiner 
{
}

-(id)init;
@end

@implementation InheritContainer

-(id)init
{
    if (self = [super initWithFoo:2 andBar:3])
    {
    }
    return self;
}

@end

int main(int argc, const char * argv[])
{

    @autoreleasepool {
        Container *j = [[Container alloc] init];
        InheritContainer *k = [[InheritContainer alloc] init];

        printf("ivar value = %u\n", (unsigned)j->_definer->foo); // Set breakpoint 0 here.
        printf("ivar value = %u\n", (unsigned)k->foo);           // Set breakpoint 1 here.
    }   
    return 0;
}

