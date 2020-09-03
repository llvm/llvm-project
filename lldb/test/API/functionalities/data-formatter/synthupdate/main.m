#import <Foundation/Foundation.h>

int main (int argc, const char * argv[])
{
    
    NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];

    NSArray* foo = [NSArray arrayWithObjects:@1,@2,@3,@4,@5, nil];
    NSDictionary *bar = @{@1 : @"one",@2 : @"two", @3 : @"three", @4 : @"four", @5 : @"five", @6 : @"six", @7 : @"seven"};
    id x = foo;
    x = bar; // Set break point at this line.

    [pool drain];
    return 0;
}

