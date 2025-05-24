#import <Foundation/Foundation.h>

@interface NSObject (Fake)
// 9 digit selector
- (void)nineDigit;
// 10 digit selector
- (void)tenDigitXX;
@end

int main() {
  SEL sel1 = @selector(nineDigit);
  NSString *str1 = NSStringFromSelector(sel1);
  SEL sel2 = @selector(tenDigitXX);
  NSString *str2 = NSStringFromSelector(sel2);
  NSLog(@"break here %@, %@", str1, str2);
  return 0;
}
