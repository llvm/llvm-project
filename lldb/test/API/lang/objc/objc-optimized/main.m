#import <Foundation/Foundation.h>

@interface MyClass : NSObject {
  int member;
}

- (id)initWithMember:(int)_member;
- (NSString*)description;
@end

@implementation MyClass

- (id)initWithMember:(int)_member
{
    if (self = [super init])
    {
      member = _member;
    }
    return self;
}

- (void)dealloc
{
    [super dealloc];
}

// Set a breakpoint on '-[MyClass description]' and test expressions: expr member
- (NSString *)description
{
    return [NSString stringWithFormat:@"%d", member];
}
@end

int main (int argc, char const *argv[])
{
    NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];

    MyClass *my_object = [[MyClass alloc] initWithMember:5];

    NSLog(@"MyObject %@", [my_object description]);

    [pool release];
    return 0;
}
