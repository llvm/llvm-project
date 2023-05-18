#import <objc/NSObject.h>
#import <stdint.h>

@interface BaseClass : NSObject
{
  int _backed_int;
#if !__OBJC2__
  int _unbacked_int;
#endif
}
@property int backed_int;
@property int unbacked_int;
@end

@interface DerivedClass : BaseClass
{
  int _derived_backed_int;
#if !__OBJC2__
  int _derived_unbacked_int;
#endif
  @public
  uint32_t flag1 : 1;
  uint32_t flag2 : 3;
}

@property int derived_backed_int;
@property int derived_unbacked_int;
@end
