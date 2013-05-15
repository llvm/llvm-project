#import <Foundation/Foundation.h>
#import <Cocoa/Cocoa.h>

int main (int argc, char const *argv[])
{
    NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];

	[NSApplication sharedApplication];
	NSWindow* window = [[NSWindow alloc] initWithContentRect:NSMakeRect(0,0,100,100) styleMask:NSBorderlessWindowMask backing:NSBackingStoreRetained defer:NO];
	[window setCanHide:YES];
    [pool release]; // Set breakpoint here.
    return 0;
}

