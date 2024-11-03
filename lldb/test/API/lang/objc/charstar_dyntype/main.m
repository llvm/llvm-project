#import <Foundation/Foundation.h>

@interface FoolMeOnce : NSObject
{
	int32_t value_one; // ivars needed to make 32-bit happy
	int32_t value_two;
}
- (FoolMeOnce *) initWithFirst: (int32_t) first andSecond: (int32_t) second;

@property int32_t value_one;
@property int32_t value_two;

@end

@implementation FoolMeOnce
@synthesize value_one;
@synthesize value_two;
- (FoolMeOnce *) initWithFirst: (int32_t) first andSecond: (int32_t) second
{
  value_one = first;
  value_two = second;
  return self;
}
@end

int main (int argc, char const *argv[])
{
    NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];

    FoolMeOnce *my_foolie = [[FoolMeOnce alloc] initWithFirst: 20 andSecond: 55];
    const char *my_string = (char *) my_foolie;

    my_string = "Now this is a REAL string..."; // Set breakpoint here.

    [pool release];
    return 0;
}
