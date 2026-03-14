@interface MyBase : NSObject 
{
#if !__OBJC2__
  int maybe_used; // The 1.0 runtime needs to have backed properties...
#endif
}
@property int propertyMovesThings;
@end
