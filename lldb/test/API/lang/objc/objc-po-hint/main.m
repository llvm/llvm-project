#import <Foundation/Foundation.h>

@interface Foo : NSObject {}

-(id) init;

@end

@implementation Foo

-(id) init
{
    return self = [super init];
}
@end

int main()
{
    Foo *foo = [Foo new];
    NSLog(@"a"); // Set breakpoint here.
    return 0;
}
