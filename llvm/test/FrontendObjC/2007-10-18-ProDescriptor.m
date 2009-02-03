// RUN: %llvmgcc -x objective-c -c %s -o /dev/null
@protocol O
@end
@interface O < O > {
}
@end
struct A {
};
@protocol AB
- (unsigned) ver;
@end
@interface AGy:O < AB > {
}
@end
@implementation AGy
- (unsigned) ver {
}
@end

