// REQUIRES: objc-gnustep
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

// RUN: %lldb -b -o "b objc-gnustep-print.m:42" -o "run" -o "p self" -o "p *self" -- %t | FileCheck %s --check-prefix=SELF
//
// SELF: (lldb) b objc-gnustep-print.m:42
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

// RUN: %lldb -b -o "b objc-gnustep-print.m:105" -o "run" -o "p t->_int" -o "p t->_float" -o "p t->_char" \
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

// LLDB resolves `_NSPrintForDebugger` by symbol in any loaded module and
// calls it to implement `po`. In a full GNUstep environment gnustep-base
// provides it; this hermetic stand-in exercises the same machinery.
const char *_NSPrintForDebugger(id object) {
  if (!object)
    return 0;
  return object_getClassName(object);
}

// A selector's name cannot be read from memory: __objc_load overwrites the
// name field with a dispatch index, so the only source is the symbol clang
// emits for it. Without a GNUstep-aware summary, Apple's SEL provider prints
// that index as a few garbage bytes in every Objective-C frame line.
//
// RUN: %lldb -b -o "b objc-gnustep-print.m:42" -o "run" -o "frame variable _cmd" \
// RUN:     -o "p _cmd" -o "expr -- (SEL *)&_cmd" -- %t | FileCheck %s --check-prefix=SEL
//
// SEL: (lldb) run
// SEL: -[TestObj check_ivars_zeroed](self={{.*}}, _cmd="check_ivars_zeroed") at objc-gnustep-print.m
//
// SEL: (lldb) frame variable _cmd
// SEL: (SEL) _cmd = 0x{{[0-9a-f]+}} "check_ivars_zeroed"
//
// SEL: (lldb) p _cmd
// SEL: (SEL) 0x{{[0-9a-f]+}} "check_ivars_zeroed"
//
// SEL: (lldb) expr -- (SEL *)&_cmd
// SEL: (SEL *) $0 = 0x{{[0-9a-f]+}} "check_ivars_zeroed"

// RUN: %lldb -b -o "b objc-gnustep-print.m:105" -o "run" -o "po t" \
// RUN:     -- %t | FileCheck %s --check-prefix=PO
//
// PO: (lldb) po t
// PO: TestObj

// Stepping at a message send goes through the objc_msgSend trampoline into
// the method implementation.
//
// RUN: %lldb -b -o "b objc-gnustep-print.m:103" -o "run" -o "step" \
// RUN:     -- %t | FileCheck %s --check-prefix=STEP
//
// STEP: (lldb) step
// STEP: stop reason = step in
// STEP: check_ivars_zeroed
