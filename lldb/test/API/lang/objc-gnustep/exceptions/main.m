#import <Foundation/Foundation.h>

static void raiser(void) {
  [[NSException exceptionWithName:@"BadThingException"
                           reason:@"the bad thing happened"
                         userInfo:nil] raise];
}

int main(int argc, const char **argv) {
  @autoreleasepool {
    NSException *caught = nil;
    @try {
      raiser();
    } @catch (NSException *e) {
      caught = e;
      NSLog(@"%@", [caught reason]); // break in handler
    }
    return caught == nil;
  }
}
