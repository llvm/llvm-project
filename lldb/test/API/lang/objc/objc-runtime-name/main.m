#import <Foundation/Foundation.h>

// This is how a Swift class is declared in a generated Objective-C header.
__attribute__((objc_runtime_name("_$s4main7RenamedC")))
@interface Renamed : NSObject
@property int value;
+ (int)classValue;
@end

@implementation Renamed
+ (int)classValue {
  return 42;
}
@end

@interface Renamed (Category)
- (int)stop;
@end

@implementation Renamed (Category)
- (int)stop {
  return self.value; // break here
}
@end

int main() {
  Renamed *r = [[Renamed alloc] init];
  r.value = 23;
  return [r stop] == 23 ? 0 : 1;
}
