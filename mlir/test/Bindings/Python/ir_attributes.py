# RUN: %PYTHON %s | FileCheck %s

import gc
from mlir.ir import *

def run(f):
  print("\nTEST:", f.__name__)
  f()
  gc.collect()
  assert Context._get_live_count() == 0


# CHECK-LABEL: TEST: testParsePrint
def testParsePrint():
  with Context() as ctx:
    t = Attribute.parse('"hello"')
  assert t.context is ctx
  ctx = None
  gc.collect()
  # CHECK: "hello"
  print(str(t))
  # CHECK: Attribute("hello")
  print(repr(t))

run(testParsePrint)


# CHECK-LABEL: TEST: testParseError
# TODO: Hook the diagnostic manager to capture a more meaningful error
# message.
def testParseError():
  with Context():
    try:
      t = Attribute.parse("BAD_ATTR_DOES_NOT_EXIST")
    except ValueError as e:
      # CHECK: Unable to parse attribute: 'BAD_ATTR_DOES_NOT_EXIST'
      print("testParseError:", e)
    else:
      print("Exception not produced")

run(testParseError)


# CHECK-LABEL: TEST: testAttrEq
def testAttrEq():
  with Context():
    a1 = Attribute.parse('"attr1"')
    a2 = Attribute.parse('"attr2"')
    a3 = Attribute.parse('"attr1"')
    # CHECK: a1 == a1: True
    print("a1 == a1:", a1 == a1)
    # CHECK: a1 == a2: False
    print("a1 == a2:", a1 == a2)
    # CHECK: a1 == a3: True
    print("a1 == a3:", a1 == a3)
    # CHECK: a1 == None: False
    print("a1 == None:", a1 == None)

run(testAttrEq)


# CHECK-LABEL: TEST: testAttrEqDoesNotRaise
def testAttrEqDoesNotRaise():
  with Context():
    a1 = Attribute.parse('"attr1"')
    not_an_attr = "foo"
    # CHECK: False
    print(a1 == not_an_attr)
    # CHECK: False
    print(a1 == None)
    # CHECK: True
    print(a1 != None)

run(testAttrEqDoesNotRaise)


# CHECK-LABEL: TEST: testAttrCapsule
def testAttrCapsule():
  with Context() as ctx:
    a1 = Attribute.parse('"attr1"')
  # CHECK: mlir.ir.Attribute._CAPIPtr
  attr_capsule = a1._CAPIPtr
  print(attr_capsule)
  a2 = Attribute._CAPICreate(attr_capsule)
  assert a2 == a1
  assert a2.context is ctx

run(testAttrCapsule)


# CHECK-LABEL: TEST: testStandardAttrCasts
def testStandardAttrCasts():
  with Context():
    a1 = Attribute.parse('"attr1"')
    astr = StringAttr(a1)
    aself = StringAttr(astr)
    # CHECK: Attribute("attr1")
    print(repr(astr))
    try:
      tillegal = StringAttr(Attribute.parse("1.0"))
    except ValueError as e:
      # CHECK: ValueError: Cannot cast attribute to StringAttr (from Attribute(1.000000e+00 : f64))
      print("ValueError:", e)
    else:
      print("Exception not produced")

run(testStandardAttrCasts)


# CHECK-LABEL: TEST: testFloatAttr
def testFloatAttr():
  with Context(), Location.unknown():
    fattr = FloatAttr(Attribute.parse("42.0 : f32"))
    # CHECK: fattr value: 42.0
    print("fattr value:", fattr.value)

    # Test factory methods.
    # CHECK: default_get: 4.200000e+01 : f32
    print("default_get:", FloatAttr.get(
        F32Type.get(), 42.0))
    # CHECK: f32_get: 4.200000e+01 : f32
    print("f32_get:", FloatAttr.get_f32(42.0))
    # CHECK: f64_get: 4.200000e+01 : f64
    print("f64_get:", FloatAttr.get_f64(42.0))
    try:
      fattr_invalid = FloatAttr.get(
          IntegerType.get_signless(32), 42)
    except ValueError as e:
      # CHECK: invalid 'Type(i32)' and expected floating point type.
      print(e)
    else:
      print("Exception not produced")

run(testFloatAttr)


# CHECK-LABEL: TEST: testIntegerAttr
def testIntegerAttr():
  with Context() as ctx:
    iattr = IntegerAttr(Attribute.parse("42"))
    # CHECK: iattr value: 42
    print("iattr value:", iattr.value)
    # CHECK: iattr type: i64
    print("iattr type:", iattr.type)

    # Test factory methods.
    # CHECK: default_get: 42 : i32
    print("default_get:", IntegerAttr.get(
        IntegerType.get_signless(32), 42))

run(testIntegerAttr)


# CHECK-LABEL: TEST: testBoolAttr
def testBoolAttr():
  with Context() as ctx:
    battr = BoolAttr(Attribute.parse("true"))
    # CHECK: iattr value: True
    print("iattr value:", battr.value)

    # Test factory methods.
    # CHECK: default_get: true
    print("default_get:", BoolAttr.get(True))

run(testBoolAttr)


# CHECK-LABEL: TEST: testStringAttr
def testStringAttr():
  with Context() as ctx:
    sattr = StringAttr(Attribute.parse('"stringattr"'))
    # CHECK: sattr value: stringattr
    print("sattr value:", sattr.value)

    # Test factory methods.
    # CHECK: default_get: "foobar"
    print("default_get:", StringAttr.get("foobar"))
    # CHECK: typed_get: "12345" : i32
    print("typed_get:", StringAttr.get_typed(
        IntegerType.get_signless(32), "12345"))

run(testStringAttr)


# CHECK-LABEL: TEST: testNamedAttr
def testNamedAttr():
  with Context():
    a = Attribute.parse('"stringattr"')
    named = a.get_named("foobar")  # Note: under the small object threshold
    # CHECK: attr: "stringattr"
    print("attr:", named.attr)
    # CHECK: name: foobar
    print("name:", named.name)
    # CHECK: named: NamedAttribute(foobar="stringattr")
    print("named:", named)

run(testNamedAttr)


# CHECK-LABEL: TEST: testDenseIntAttr
def testDenseIntAttr():
  with Context():
    raw = Attribute.parse("dense<[[0,1,2],[3,4,5]]> : vector<2x3xi32>")
    # CHECK: attr: dense<[{{\[}}0, 1, 2], [3, 4, 5]]>
    print("attr:", raw)

    a = DenseIntElementsAttr(raw)
    assert len(a) == 6

    # CHECK: 0 1 2 3 4 5
    for value in a:
      print(value, end=" ")
    print()

    # CHECK: i32
    print(ShapedType(a.type).element_type)

    raw = Attribute.parse("dense<[true,false,true,false]> : vector<4xi1>")
    # CHECK: attr: dense<[true, false, true, false]>
    print("attr:", raw)

    a = DenseIntElementsAttr(raw)
    assert len(a) == 4

    # CHECK: 1 0 1 0
    for value in a:
      print(value, end=" ")
    print()

    # CHECK: i1
    print(ShapedType(a.type).element_type)


run(testDenseIntAttr)


# CHECK-LABEL: TEST: testDenseFPAttr
def testDenseFPAttr():
  with Context():
    raw = Attribute.parse("dense<[0.0, 1.0, 2.0, 3.0]> : vector<4xf32>")
    # CHECK: attr: dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00]>

    print("attr:", raw)

    a = DenseFPElementsAttr(raw)
    assert len(a) == 4

    # CHECK: 0.0 1.0 2.0 3.0
    for value in a:
      print(value, end=" ")
    print()

    # CHECK: f32
    print(ShapedType(a.type).element_type)


run(testDenseFPAttr)


# CHECK-LABEL: TEST: testTypeAttr
def testTypeAttr():
  with Context():
    raw = Attribute.parse("vector<4xf32>")
    # CHECK: attr: vector<4xf32>
    print("attr:", raw)
    type_attr = TypeAttr(raw)
    # CHECK: f32
    print(ShapedType(type_attr.value).element_type)


run(testTypeAttr)
