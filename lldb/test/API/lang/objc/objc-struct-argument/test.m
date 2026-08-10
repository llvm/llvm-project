#import <Foundation/Foundation.h>
#include <TargetConditionals.h>

#if TARGET_OS_IPHONE
@import CoreGraphics;
typedef CGRect NSRect;
#endif

struct things_to_sum {
    int a;
    int b;
    int c;
};

@interface ThingSummer : NSObject {
};
-(int)sumThings:(struct things_to_sum)tts;
@end

@implementation ThingSummer
-(int)sumThings:(struct things_to_sum)tts
{
  return tts.a + tts.b + tts.c;
}
@end

int main()
{
  @autoreleasepool
  {
    ThingSummer *summer = [ThingSummer alloc];
    struct things_to_sum tts = { 2, 3, 4 };
    int ret = [summer sumThings:tts];
    NSRect rect = {{0, 0}, {10, 20}};    
	// The Objective-C V1 runtime won't read types from metadata so we need
	// NSValue in our debug info to use it in our test.
	NSValue *v = [NSValue valueWithRect:rect];
    return rect.origin.x; // Set breakpoint here.
  }
}
