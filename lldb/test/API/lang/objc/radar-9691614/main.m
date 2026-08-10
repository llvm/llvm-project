#import <Foundation/Foundation.h>
#include <stdio.h>

@interface MyString : NSObject {
    NSString *str;
    NSDate *date;
    BOOL _is_valid;
}

- (id)initWithNSString:(NSString *)string;
- (BOOL)isValid;
@end

@implementation MyString
- (id)initWithNSString:(NSString *)string
{
    if (self = [super init])
    {
        str = [NSString stringWithString:string];
        date = [NSDate date];
    }
    _is_valid = YES;
    return self;
}

- (BOOL)isValid
{
    return _is_valid;
}

- (void)dealloc
{
    [date release];
    [str release];
    [super dealloc];
}

- (NSString *)description
{
    return [str stringByAppendingFormat:@" with timestamp: %@", date];
}
@end

void
Test_MyString (const char *program)
{
    NSString *str = [NSString stringWithFormat:@"Hello from '%s'", program];
    MyString *my = [[MyString alloc] initWithNSString:str];
    if ([my isValid])
        printf("my is valid!\n");

    NSLog(@"NSString instance: %@", [str description]); // Set breakpoint here.
                                                        // Test 'p (int)[my isValid]'.
                                                        // The expression parser should not crash -- rdar://problem/9691614.

    NSLog(@"MyString instance: %@", [my description]);
}

int main (int argc, char const *argv[])
{
    NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];

    Test_MyString (argv[0]);

    [pool release];
    return 0;
}
