#import <Foundation/Foundation.h>
#if __has_include(<AppKit/AppKit.h>)
#import <AppKit/AppKit.h>
#define XXView NSView
#else
#import <UIKit/UIKit.h>
#define XXView UIView
#endif

int main() {
  XXView *view = [[XXView alloc] init];
  dispatch_group_t g = dispatch_group_create();
  dispatch_group_enter(g);
  [NSThread detachNewThreadWithBlock:^{
    @autoreleasepool {
      [view superview];
    }
    dispatch_group_leave(g);
  }];
  dispatch_group_wait(g, DISPATCH_TIME_FOREVER);
}
