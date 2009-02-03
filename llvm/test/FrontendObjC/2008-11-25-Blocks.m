// RUN: %llvmgcc -c %s -o /dev/null
// rdar://6394879

@interface bork
- (id)B:(void (^)())blk;
- (void)C;
@end
@implementation bork
- (id)B:(void (^)())blk {
  __attribute__((__blocks__(byref))) bork* new = ((void *)0);
  blk();
}
- (void)C {
  __attribute__((__blocks__(byref))) id var;
  [self B:^() {}];
}
@end
