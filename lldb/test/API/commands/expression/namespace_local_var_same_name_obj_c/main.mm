#import <objc/NSObject.h>
@interface Util : NSObject
+ (void)debugPrintErrorStatic;
- (void)debugPrintError;
@end

int main(int argc, const char * argv[]) {
  [Util debugPrintErrorStatic];

  Util *u = [[Util alloc] init];

  [u debugPrintError];

  return 0;
}

