// REQUIRES: objc-gnustep
// XFAIL: system-windows
//
// RUN: %build %s --compiler=clang --objc-gnustep --output=%t

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
  int refcount;
}
@end
@implementation NSObject
- (id)class {
  return object_getClass(self);
}
+ (id)new {
  return class_createInstance(self, 0);
}
@end

@interface TestObj : NSObject {
  int _int;
  float _float;
  char _char;
  void *_ptr_void;
  NSObject *_ptr_nsobject;
  id _id_objc;
}
- (void)check_ivars_zeroed;
- (void)set_ivars;
@end
@implementation TestObj
- (void)check_ivars_zeroed {
  ;
}
- (void)set_ivars {
  _int = 1;
  _float = 2.0f;
  _char = '\3';
  _ptr_void = (void*)4;
  _ptr_nsobject = (NSObject*)5;
  _id_objc = (id)6;
}
@end

// RUN: %lldb -b -o "b objc-gnustep-print.m:43" -o "run" -o "p self" -o "p *self" -- %t | FileCheck %s --check-prefix=SELF
//
// SELF: (lldb) b objc-gnustep-print.m:43
// SELF: Breakpoint {{.*}} at objc-gnustep-print.m
//
// SELF: (lldb) run
// SELF: Process {{[0-9]+}} stopped
// SELF: -[TestObj check_ivars_zeroed](self=[[SELF_PTR:0x[0-9a-f]+]]{{.*}}) at objc-gnustep-print.m
//
// SELF: (lldb) p self
// SELF: (TestObj *) [[SELF_PTR]]
//
// SELF: (lldb) p *self
// SELF: (TestObj) {
// SELF:   NSObject = {
// SELF:     isa
// SELF:     refcount
// SELF:   }
// SELF:   _int = 0
// SELF:   _float = 0
// SELF:   _char = '\0'
// SELF:   _ptr_void = 0x{{0*}}
// SELF:   _ptr_nsobject = nil
// SELF:   _id_objc = nil
// SELF: }

// RUN: %lldb -b -o "b objc-gnustep-print.m:106" -o "run" -o "p t->_int" -o "p t->_float" -o "p t->_char" \
// RUN:          -o "p t->_ptr_void" -o "p t->_ptr_nsobject" -o "p t->_id_objc" -- %t | FileCheck %s --check-prefix=IVARS_SET
//
// IVARS_SET: (lldb) p t->_int
// IVARS_SET: (int) 1
//
// IVARS_SET: (lldb) p t->_float
// IVARS_SET: (float) 2
//
// IVARS_SET: (lldb) p t->_char
// IVARS_SET: (char) '\x03'
//
// IVARS_SET: (lldb) p t->_ptr_void
// IVARS_SET: (void *) 0x{{0*}}4
//
// IVARS_SET: (lldb) p t->_ptr_nsobject
// IVARS_SET: (NSObject *) 0x{{0*}}5
//
// IVARS_SET: (lldb) p t->_id_objc
// IVARS_SET: (id) 0x{{0*}}6

int main() {
  TestObj *t = [TestObj new];
  [t check_ivars_zeroed];
  [t set_ivars];
  return 0;
}
