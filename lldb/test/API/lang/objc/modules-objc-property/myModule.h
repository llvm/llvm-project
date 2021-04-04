#ifndef MYMODULE
#define MYMODULE

@import ObjectiveC;

@interface MyClass : NSObject
- (int) propConflict;
+ (int) propConflict;
@property(readonly) int propConflict;
@property(readonly,class) int propConflict;
@end

@implementation MyClass
- (int) propConflict
{
  return 5;
}
+ (int) propConflict
{
  return 6;
}
@end

#endif // MYMODULE
