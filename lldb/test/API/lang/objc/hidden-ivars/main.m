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
@property (nonatomic, strong) NSMutableArray *filteredDataSource;
-(id)init;
@end

@implementation InheritContainer

-(id)init
{
    if (self = [super initWithFoo:2 andBar:3])
    {
        self.filteredDataSource = [NSMutableArray arrayWithObjects:@"hello", @"world", nil];
    }
    return self;
}

@end

int main(int argc, const char * argv[])
{
    @autoreleasepool {
        Container *j = [[Container alloc] init];
        InheritContainer *k = [[InheritContainer alloc] init];

        printf("ivar value = %u\n", (unsigned)j->_definer->foo); // breakpoint1
        printf("ivar value = %u\n", (unsigned)k->foo);
    }   
    return 0;
}

