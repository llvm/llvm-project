// Compiled without debug information, so the only description of Hidden
// that reaches the debugger is libobjc2's own runtime metadata. The header
// this file's users see declares nothing but the class name.

#import "objc/runtime.h"

@protocol NSCoding
@end

#ifdef __has_attribute
#if __has_attribute(objc_root_class)
__attribute__((objc_root_class))
#endif
#endif
@interface NSObject <NSCoding> {
  id isa;
}
+ (id)new;
@end
@implementation NSObject
+ (id)new {
  return class_createInstance(self, 0);
}
@end

// The extended method encoding spells an object parameter with its class
// name, so a digit in the name lands in the middle of the type string.
@interface SHA256 : NSObject
@end
@implementation SHA256
@end

@interface Hidden : NSObject {
@public
  int _int;
  float _float;
  char _char;
  void *_ptr;
}
- (int)doubled:(int)value;
- (int)plainInt;
- (int)digest:(SHA256 *)hash;
+ (int)classAnswer;
@end
@implementation Hidden
- (int)doubled:(int)value {
  return value * 2;
}
- (int)plainInt {
  return _int;
}
- (int)digest:(SHA256 *)hash {
  return hash == 0 ? 5 : 6;
}
+ (int)classAnswer {
  return 7;
}
@end

id MakeHidden(void) {
  Hidden *hidden = [Hidden new];
  hidden->_int = 1;
  hidden->_float = 2.0f;
  hidden->_char = '\3';
  hidden->_ptr = (void *)4;
  return hidden;
}
