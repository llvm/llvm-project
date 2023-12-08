@import Umbrella;

@interface Bar : Foo
+(instancetype)init;
@end

@implementation Bar
+(instancetype)init {
  return [super init];
}
@end

int main(int argc, char **argv) {
  id bar = [Bar new];
  [bar i]; // break here
  return 0;
}
