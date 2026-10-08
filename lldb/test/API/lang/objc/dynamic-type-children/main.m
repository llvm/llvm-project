#import <Foundation/Foundation.h>
#if defined (__i386__) || defined (__x86_64__)
#import <Cocoa/Cocoa.h>
#else
#import <UIKit/UIKit.h>
#endif

int main (int argc, char const *argv[])
{
    NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];
#if defined (__i386__) || defined (__x86_64__)

    [NSApplication sharedApplication];
    NSWindow* window = [[NSWindow alloc] initWithContentRect:NSMakeRect(0,0,100,100) styleMask:NSBorderlessWindowMask backing:NSBackingStoreRetained defer:NO];
    [window setCanHide:YES];
#else
    [UIApplication sharedApplication];
    CGRect rect = { 0, 0, 100, 100};
    UIWindow* window = [[UIWindow alloc] initWithFrame:rect];
#endif
    [pool release]; // Set breakpoint here.
    return 0;
}

