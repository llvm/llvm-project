#import <Foundation/Foundation.h>

int main ()
{
    @autoreleasepool
    {
        struct foo {
            int x;
            int y;
        } myFoo;

        myFoo.x = 2;
        myFoo.y = 3;    // First breakpoint

        NSString *string = [NSString stringWithFormat:@"%s", "Hello world!"];

        NSLog(@"%d %@", myFoo.x, string); // Second breakpoint
    }
}
