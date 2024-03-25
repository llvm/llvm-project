// RUN: llvm-tblgen -I %p/../../../../include -gen-global-isel-combiner \
// RUN:     -combiners=Combiner %s 2>&1 | FileCheck %s

include "llvm/Target/Target.td"
include "llvm/Target/GlobalISel/Combine.td"

// Check we don't crash if a combine is present twice in the list.

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

def dummy;

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: warning: skipping rule 'Foo' because it has already been processed
def Foo : GICombineRule<
  (defs root:$root),
  (match (G_ZEXT $root, $x)),
  (apply (G_TRUNC $root, $x))>;

def Bar : GICombineRule<
  (defs root:$root),
  (match (G_TRUNC $root, $x)),
  (apply (G_ZEXT $root, $x))>;

def FooBar : GICombineGroup<[ Foo, Bar ]>;

def Combiner: GICombiner<"GenMyCombiner", [
  FooBar,
  Foo
]>;
