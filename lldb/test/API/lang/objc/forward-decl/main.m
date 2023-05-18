#import <objc/NSObject.h>
#import "Container.h"
#import <stdio.h>

int main(int argc, const char * argv[])
{

    @autoreleasepool {
        Container *j = [[Container alloc] init];

        printf("member value = %p", [j getMember]); // Set breakpoint 0 here.
    }   
    return 0;
}

