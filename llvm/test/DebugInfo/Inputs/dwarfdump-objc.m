// Compile with clang -g dwarfdump-objc.m -c -Wno-objc-root-class

@interface NSObject {} @end


@interface TestInterface
@property (readonly) int ReadOnly;
@property (assign) int Assign;
@property (readwrite) int ReadWrite;
@property (retain) NSObject *Retain;
@property (copy) NSObject *Copy;
@property (nonatomic) int NonAtomic;
@property (atomic) int Atomic;
@property (strong) NSObject *Strong;
@property (unsafe_unretained) id UnsafeUnretained;
@property (nullable) NSObject *Nullability;
@property (null_resettable) NSObject *NullResettable;
@property (class) int ClassProperty;
@end

@implementation TestInterface
@end
