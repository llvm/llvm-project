#import <Foundation/Foundation.h>

int main(int argc, char const *argv[]) {
  @try {
    NSException *e = [[NSException alloc] initWithName:@"ThrownException"
                                      reason:@"SomeReason"
                                    userInfo:nil];
    @throw e;
  } @catch (NSException *e) {
    NSLog(@"Caught %@", e);
    @throw; // let the process crash...
  }
  return 0;
}
