#import <Foundation/Foundation.h>

int main (int argc, const char * argv[])
{
    
    NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];

	OSType  a = 'test';
	OSType b = 'best';
	
    [pool drain];// Set break point at this line.
    return 0;
}

