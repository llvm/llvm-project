// RUN: %llvmgcc -c %s -fobjc-gc -o /dev/null
typedef int NSInteger;
typedef struct _NSRect {
  int origin;
  int size;
} NSRect;

__attribute__((objc_gc(strong))) NSRect *_cachedRectArray;
extern const NSRect NSZeroRect;
@interface A{
}
-(void)bar:(NSInteger *)rectCount;
@end

@implementation A 

-(void)bar:(NSInteger *)rectCount {
  NSRect appendRect = NSZeroRect; 

  _cachedRectArray[*rectCount - 1] = NSZeroRect; 
}

@end
