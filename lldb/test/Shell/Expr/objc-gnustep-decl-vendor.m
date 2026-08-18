// REQUIRES: objc-gnustep
//
// A class whose only description is libobjc2's runtime metadata: the other
// half of this program is compiled with -g0, and nothing here declares
// anything about Hidden beyond its name.
//
// RUN: %build %S/Inputs/objc-gnustep-hidden-class.m --compiler=clang --objc-gnustep \
// RUN:     --no-debug-info --mode=compile --output=%t-hidden.o
// RUN: %build %s --compiler=clang --objc-gnustep --mode=compile --output=%t-main.o
// RUN: %build %t-hidden.o %t-main.o --compiler=clang --objc-gnustep --mode=link --output=%t

@class Hidden;
id MakeHidden(void);

int main() {
  Hidden *hidden = (Hidden *)MakeHidden();
  return hidden != 0; // break here
}

// The premise: the debug info really does not describe this class, so
// `image lookup -t` finds nothing and exits non-zero. If this ever starts
// matching, the rest of the test proves nothing.
//
// RUN: not %lldb -b -o "image lookup -t Hidden" -- %t \
// RUN:     | FileCheck %s --check-prefix=NODWARF
//
// NODWARF-NOT: name = "Hidden"

// The interface is synthesized from the runtime instead, so the type can be
// named and its ivars read.
//
// RUN: %lldb -b -o "b objc-gnustep-decl-vendor.m:17" -o "run" \
// RUN:     -o "type lookup Hidden" \
// RUN:     -o "expr -- ((Hidden *)hidden)->_int" \
// RUN:     -o "expr -- ((Hidden *)hidden)->_float" \
// RUN:     -o "frame variable -d run-target *hidden" \
// RUN:     -o "expr -- *(Hidden *)hidden" \
// RUN:     -- %t | FileCheck %s --check-prefix=VENDOR
//
// VENDOR: (lldb) type lookup Hidden
// VENDOR: @interface Hidden
// VENDOR-DAG: int _int;
// VENDOR-DAG: float _float;
// VENDOR-DAG: char _char;
//
// VENDOR: (lldb) expr -- ((Hidden *)hidden)->_int
// VENDOR: (int) $0 = 1
//
// VENDOR: (lldb) expr -- ((Hidden *)hidden)->_float
// VENDOR: (float) $1 = 2
//
// And the object expands in full, which is what a debugger UI shows.
//
// VENDOR: (lldb) frame variable -d run-target *hidden
// VENDOR-DAG: _int = 1
// VENDOR-DAG: _float = 2
// VENDOR-DAG: _char = '{{.*}}3'
// VENDOR-DAG: _ptr = 0x{{0*}}4
//
// The vendor supplies a *complete* type, not just a name, so dereferencing
// through it works in an expression and not only in `frame variable`.
//
// VENDOR: (lldb) expr -- *(Hidden *)hidden
// VENDOR: (Hidden) {{\$[0-9]+}} = {
// VENDOR-DAG: _int = 1
// VENDOR-DAG: _float = 2
