// RUN: %llvmgcc -x objective-c -m64 -S %s -o - | grep {L_unnamed_cfstring_}

@class NSString;

@interface A
- (void)bork:(NSString*)msg;
@end

void func(A *a) {
  [a bork:@"Hello world!"];
}
