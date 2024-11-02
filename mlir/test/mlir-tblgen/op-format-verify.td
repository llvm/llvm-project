// RUN: mlir-tblgen -gen-op-decls -asmformat-error-is-fatal=false -I %S/../../include %s -o=%t 2>&1 | FileCheck %s

include "mlir/IR/OpBase.td"

def TestDialect : Dialect {
  let name = "test";
}
class TestFormat_Op<string fmt, list<Trait> traits = []>
    : Op<TestDialect, "format_op", traits> {
  let assemblyFormat = fmt;
}

//===----------------------------------------------------------------------===//
// Format ambiguity caused by attribute followed by colon literal
//===----------------------------------------------------------------------===//

// Test attribute followed by a colon.
// CHECK: error: format ambiguity caused by `:` literal found after attribute `attr`
def AmbiguousTypeA : TestFormat_Op<[{
  $attr `:` attr-dict
}]>, Arguments<(ins AnyAttr:$attr)>;

// Test optional attribute followed by colon.
// CHECK: error: format ambiguity caused by `:` literal found after attribute `attr`
def AmbiguousTypeB : TestFormat_Op<[{
  (`foo` $attr^)? `:` attr-dict
}]>, Arguments<(ins OptionalAttr<AnyAttr>:$attr)>;

// Test attribute followed by whitespace and then colon.
// CHECK: error: format ambiguity caused by `:` literal found after attribute `attr`
def AmbiguousTypeC : TestFormat_Op<[{
  $attr ` ` `:` attr-dict
}]>, Arguments<(ins AnyAttr:$attr)>;

// Test attribute followed by optional dictionary and then colon.
// CHECK: error: format ambiguity caused by `:` literal found after attribute `attr`
def AmbiguousTypeD : TestFormat_Op<[{
  $attr attr-dict `:`
}]>, Arguments<(ins AnyAttr:$attr)>;

// Test attribute followed by optional group and then colon.
// CHECK: error: format ambiguity caused by `:` literal found after attribute `attr`
def AmbiguousTypeE : TestFormat_Op<[{
  $attr ($a^)? `:` attr-dict type($a)
}]>, Arguments<(ins AnyAttr:$attr, Optional<I32>:$a)>;

// Test attribute followed by optional group with literals and then colon.
// CHECK: error: format ambiguity caused by `:` literal found after attribute `attr`
def AmbiguousTypeF : TestFormat_Op<[{
  $attr (`(` $a^ `)`)? `:` attr-dict (`(` type($a)^ `)`)?
}]>, Arguments<(ins AnyAttr:$attr, Optional<I32>:$a)>;

// Test attribute followed by optional group with else group.
// CHECK: error: format ambiguity caused by `:` literal found after attribute `attr`
def AmbiguousTypeG : TestFormat_Op<[{
  $attr (`(` $a^ `)`) : (`foo`)? `:` attr-dict (`(` type($a)^ `)`)?
}]>, Arguments<(ins AnyAttr:$attr, Optional<I32>:$a)>;

// Test attribute followed by optional group with colon.
// CHECK: error: format ambiguity caused by `:` literal found after attribute `attr`
def AmbiguousTypeH : TestFormat_Op<[{
  $attr (`:` $a^ `)`)? attr-dict (`(` type($a)^ `)`)?
}]>, Arguments<(ins AnyAttr:$attr, Optional<I32>:$a)>;

// Test attribute followed by optional group with colon in else group.
// CHECK: error: format ambiguity caused by `:` literal found after attribute `attr`
def AmbiguousTypeI : TestFormat_Op<[{
  $attr (`(` $a^ `)`) : (`:`)? attr-dict (`(` type($a)^ `)`)?
}]>, Arguments<(ins AnyAttr:$attr, Optional<I32>:$a)>;

// Test attribute followed by two optional groups and then a colon.
// CHECK: error: format ambiguity caused by `:` literal found after attribute `attr`
def AmbiguousTypeJ : TestFormat_Op<[{
  $attr (`(` $a^ type($a) `)`) : (`foo`)? ` ` attr-dict (`(` $b^ type($b) `)`)?
  `:`
}], [AttrSizedOperandSegments]>,
  Arguments<(ins AnyAttr:$attr, Optional<I32>:$a, Optional<I32>:$b)>;

// Test attribute followed by two optional groups and then a colon in the else
// group.
// CHECK: error: format ambiguity caused by `:` literal found after attribute `attr`
def AmbiguousTypeK : TestFormat_Op<[{
  $attr (`(` $a^ type($a) `)`) : (`foo`)? ` ` attr-dict
  (`(` $b^ type($b) `)`) : (`:`)?
}], [AttrSizedOperandSegments]>,
  Arguments<(ins AnyAttr:$attr, Optional<I32>:$a, Optional<I32>:$b)>;

// Test attribute followed by two optional groups with guarded colons but then a
// colon.
// CHECK: error: format ambiguity caused by `:` literal found after attribute `attr`
def AmbiguousTypeL : TestFormat_Op<[{
  $attr (`(` $a^ `:` type($a) `)`) : (`foo` `:`)? ` ` attr-dict
  (`(` $b^ `:` type($b) `)`) : (`foo` `:`)? `:`
}], [AttrSizedOperandSegments]>,
  Arguments<(ins AnyAttr:$attr, Optional<I32>:$a, Optional<I32>:$b)>;

// Test optional attribute followed by optional groups with a colon along one
// path.
// CHECK: error: format ambiguity caused by `:` literal found after attribute `attr`
def AmbiguousTypeM : TestFormat_Op<[{
  (`(` $attr^ ` `)? (`(` $a^ `:` type($a) `)`) : (`foo` `:`)? ` ` attr-dict
  (`(` $b^ `:` type($b) `)`) : (`foo` `:`)? `:`
}], [AttrSizedOperandSegments]>,
  Arguments<(ins OptionalAttr<AnyAttr>:$attr, Optional<I32>:$a,
                 Optional<I32>:$b)>;

// Test optional attribute followed by optional groups with a colon along one
// path inside an optional group.
// CHECK: error: format ambiguity caused by `:` literal found after attribute `attr`
def AmbiguousTypeN : TestFormat_Op<[{
  (`(` $attr^ ` `)? (`(` $a^ `:` type($a) `)`) : (`foo` `:`)? ` ` attr-dict
  (`(` $b^ `:` type($b) `)`) : (`:`)?
}], [AttrSizedOperandSegments]>,
  Arguments<(ins OptionalAttr<AnyAttr>:$attr, Optional<I32>:$a,
                 Optional<I32>:$b)>;

// Test attribute followed by optional attribute, operand, successor, region,
// and a colon.
// CHECK: error: format ambiguity caused by `:` literal found after attribute `attr`
def AmbiguousTypeO : TestFormat_Op<[{
  $attr attr-dict $a $b $c $d $e `:`
}], [AttrSizedOperandSegments]> {
  let arguments = (ins AnyAttr:$attr, OptionalAttr<I32Attr>:$a,
                       Optional<I32>:$b, Variadic<I32>:$c);
  let successors = (successor VariadicSuccessor<AnySuccessor>:$d);
  let regions = (region VariadicRegion<AnyRegion>:$e);
}

// Test two attributes, where the second one is ambiguous.
// CHECK: error: format ambiguity caused by `:` literal found after attribute `b`
def AmbiguousTypeP : TestFormat_Op<[{
  $a attr-dict `(` `:` $b (`:` $c^)?
}]>, Arguments<(ins AnyAttr:$a, AnyAttr:$b, Optional<I32>:$c)>;

// Test two attributes, where the second one is ambiguous.
// CHECK: error: format ambiguity caused by `:` literal found after attribute `b`
def AmbiguousTypeQ : TestFormat_Op<[{
  $a attr-dict (`(` $c^ `:`)? `(` `:` $b `:`
}]>, Arguments<(ins AnyAttr:$a, AnyAttr:$b, Optional<I32>:$c)>;

// CHECK-NOT: error

// Test attribute followed by two optional groups with guarded colons.
def ValidTypeA : TestFormat_Op<[{
  $attr (`(` $a^ `:` type($a) `)`) : (`foo` `:`)? ` ` attr-dict
  (`(` $b^ `:` type($b) `)`) : (`foo` `:`)? ` ` `(` `:`
}], [AttrSizedOperandSegments]>,
  Arguments<(ins AnyAttr:$attr, Optional<I32>:$a, Optional<I32>:$b)>;

// Test optional attribute followed by two optional groups with guarded colons.
def ValidTypeB : TestFormat_Op<[{
  (`(` $attr^ ` `)? (`(` $a^ `:` type($a) `)`) : (`foo` `:`)? ` ` attr-dict
  (`(` $b^ `:` type($b) `)`) : (`foo` `:`)? ` ` `(` `:`
}], [AttrSizedOperandSegments]>,
  Arguments<(ins OptionalAttr<AnyAttr>:$attr, Optional<I32>:$a,
                 Optional<I32>:$b)>;

// Test optional attribute guarded colon along within segment.
def ValidTypeC : TestFormat_Op<[{
  (`(` $attr^ `)`) : (`:`)? attr-dict `:`
}]>, Arguments<(ins OptionalAttr<AnyAttr>:$attr)>;

// Test optional group guard blocks colon.
def ValidTypeD : TestFormat_Op<[{
  $a attr-dict ($c^ `:`)?
}]>, Arguments<(ins AnyAttr:$a, Optional<I32>:$c)>;
