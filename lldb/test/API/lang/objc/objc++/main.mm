#include <Foundation/NSObject.h>

@interface F : NSObject
@end

@implementation F
{
@public
    int f;
}

@end

int main(int argc, char* argv[])
{
    F* f = [F new];
    f->f = 3;
    return 0; // breakpoint 1
}
