// Compile with clang -g dwarfdump-objc.m -c -Wno-objc-root-class

@interface NSObject {} @end


@interface TestInterface
@property (readonly) int ReadOnly;
@property (assign) int Assign;
@property (readwrite) int ReadWrite;
@property (retain) NSObject *Retain;
@property (copy) NSObject *Copy;
@property (nonatomic) int NonAtomic;
@end

@implementation TestInterface
@end
