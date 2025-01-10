#import <Foundation/Foundation.h>

int main(int argc, char const *argv[]) {
  @throw [[NSException alloc] initWithName:@"ThrownException"
                                    reason:@"SomeReason"
                                  userInfo:nil];
  return 0;
}
