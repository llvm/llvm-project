#import "Container.h"

@interface ForwardDeclaredClass : NSObject
{
    int a;
    int b;
}
@end

@implementation ForwardDeclaredClass

@end

@implementation Container

-(id)init
{
    member = [ForwardDeclaredClass alloc];
    return [super init];
}

-(ForwardDeclaredClass *)getMember
{
    return member;
}

@end
