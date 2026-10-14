// RUN: rm -rf %t && split-file %s %t
// RUN: not llvm-tblgen -gen-sd-node-info -I %p/../../../include \
// RUN:   %t/tsflags.td -o /dev/null 2>&1 | FileCheck %t/tsflags.td
// RUN: not llvm-tblgen -gen-sd-node-info -I %p/../../../include \
// RUN:   %t/name-offset.td -o /dev/null 2>&1 | FileCheck %t/name-offset.td
// RUN: not llvm-tblgen -gen-sd-node-info -I %p/../../../include \
// RUN:   %t/result-count.td -o /dev/null 2>&1 | FileCheck %t/result-count.td
// RUN: not llvm-tblgen -gen-sd-node-info -I %p/../../../include \
// RUN:   %t/operand-count.td -o /dev/null 2>&1 | FileCheck %t/operand-count.td

//--- tsflags.td
include "llvm/Target/Target.td"

def MyTarget : Target;

let TSFlags = 256 in
def bad : SDNode<"MyTargetISD::BAD", SDTypeProfile<0, 0, []>>;
// CHECK: error: SDNode TSFlags value 256 does not fit in 8 bits

//--- name-offset.td
include "llvm/Target/Target.td"

def MyTarget : Target;

defvar S2 = !strconcat("x", "x");
defvar S4 = !strconcat(S2, S2);
defvar S8 = !strconcat(S4, S4);
defvar S16 = !strconcat(S8, S8);
defvar S32 = !strconcat(S16, S16);
defvar S64 = !strconcat(S32, S32);
defvar S128 = !strconcat(S64, S64);
defvar S256 = !strconcat(S128, S128);
defvar S512 = !strconcat(S256, S256);
defvar S1024 = !strconcat(S512, S512);
defvar S2048 = !strconcat(S1024, S1024);
defvar S4096 = !strconcat(S2048, S2048);
defvar S8192 = !strconcat(S4096, S4096);
defvar S16384 = !strconcat(S8192, S8192);
defvar S32768 = !strconcat(S16384, S16384);
defvar S65536 = !strconcat(S32768, S32768);

def first :
    SDNode<!strconcat("MyTargetISD::A", S65536), SDTypeProfile<0, 0, []>>;
def bad : SDNode<"MyTargetISD::Z", SDTypeProfile<0, 0, []>>;
// CHECK: error: generated SDNode name offset {{[0-9]+}} does not fit in 16 bits

//--- result-count.td
include "llvm/Target/Target.td"

def MyTarget : Target;

def bad : SDNode<"MyTargetISD::BAD", SDTypeProfile<65536, 0, []>>;
// CHECK: error: SDNode result count 65536 does not fit in 16 bits

//--- operand-count.td
include "llvm/Target/Target.td"

def MyTarget : Target;

def bad : SDNode<"MyTargetISD::BAD", SDTypeProfile<0, 32768, []>>;
// CHECK: error: SDNode operand count 32768 does not fit in 16 bits
