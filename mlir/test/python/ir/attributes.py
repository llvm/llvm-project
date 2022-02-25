# RUN: %PYTHON %s | FileCheck %s

import gc
from mlir.ir import *

def run(f):
  print("\nTEST:", f.__name__)
  f()
  gc.collect()
  assert Context._get_live_count() == 0
  return f


# CHECK-LABEL: TEST: testParsePrint
@run
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


# CHECK-LABEL: TEST: testParseError
# TODO: Hook the diagnostic manager to capture a more meaningful error
# message.
@run
def testParseError():
  with Context():
    try:
      t = Attribute.parse("BAD_ATTR_DOES_NOT_EXIST")
    except ValueError as e:
      # CHECK: Unable to parse attribute: 'BAD_ATTR_DOES_NOT_EXIST'
      print("testParseError:", e)
    else:
      print("Exception not produced")


# CHECK-LABEL: TEST: testAttrEq
@run
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


# CHECK-LABEL: TEST: testAttrHash
@run
def testAttrHash():
  with Context():
    a1 = Attribute.parse('"attr1"')
    a2 = Attribute.parse('"attr2"')
    a3 = Attribute.parse('"attr1"')
    # CHECK: hash(a1) == hash(a3): True
    print("hash(a1) == hash(a3):", a1.__hash__() == a3.__hash__())

    s = set()
    s.add(a1)
    s.add(a2)
    s.add(a3)
    # CHECK: len(s): 2
    print("len(s): ", len(s))


# CHECK-LABEL: TEST: testAttrCast
@run
def testAttrCast():
  with Context():
    a1 = Attribute.parse('"attr1"')
    a2 = Attribute(a1)
    # CHECK: a1 == a2: True
    print("a1 == a2:", a1 == a2)


# CHECK-LABEL: TEST: testAttrIsInstance
@run
def testAttrIsInstance():
  with Context():
    a1 = Attribute.parse("42")
    a2 = Attribute.parse("[42]")
    assert IntegerAttr.isinstance(a1)
    assert not IntegerAttr.isinstance(a2)
    assert not ArrayAttr.isinstance(a1)
    assert ArrayAttr.isinstance(a2)


# CHECK-LABEL: TEST: testAttrEqDoesNotRaise
@run
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


# CHECK-LABEL: TEST: testAttrCapsule
@run
def testAttrCapsule():
  with Context() as ctx:
    a1 = Attribute.parse('"attr1"')
  # CHECK: mlir.ir.Attribute._CAPIPtr
  attr_capsule = a1._CAPIPtr
  print(attr_capsule)
  a2 = Attribute._CAPICreate(attr_capsule)
  assert a2 == a1
  assert a2.context is ctx


# CHECK-LABEL: TEST: testStandardAttrCasts
@run
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


# CHECK-LABEL: TEST: testAffineMapAttr
@run
def testAffineMapAttr():
  with Context() as ctx:
    d0 = AffineDimExpr.get(0)
    d1 = AffineDimExpr.get(1)
    c2 = AffineConstantExpr.get(2)
    map0 = AffineMap.get(2, 3, [])

    # CHECK: affine_map<(d0, d1)[s0, s1, s2] -> ()>
    attr_built = AffineMapAttr.get(map0)
    print(str(attr_built))

    attr_parsed = Attribute.parse(str(attr_built))
    assert attr_built == attr_parsed


# CHECK-LABEL: TEST: testFloatAttr
@run
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


# CHECK-LABEL: TEST: testIntegerAttr
@run
def testIntegerAttr():
  with Context() as ctx:
    i_attr = IntegerAttr(Attribute.parse("42"))
    # CHECK: i_attr value: 42
    print("i_attr value:", i_attr.value)
    # CHECK: i_attr type: i64
    print("i_attr type:", i_attr.type)
    si_attr = IntegerAttr(Attribute.parse("-1 : si8"))
    # CHECK: si_attr value: -1
    print("si_attr value:", si_attr.value)
    ui_attr = IntegerAttr(Attribute.parse("255 : ui8"))
    # CHECK: ui_attr value: 255
    print("ui_attr value:", ui_attr.value)
    idx_attr = IntegerAttr(Attribute.parse("-1 : index"))
    # CHECK: idx_attr value: -1
    print("idx_attr value:", idx_attr.value)

    # Test factory methods.
    # CHECK: default_get: 42 : i32
    print("default_get:", IntegerAttr.get(
        IntegerType.get_signless(32), 42))


# CHECK-LABEL: TEST: testBoolAttr
@run
def testBoolAttr():
  with Context() as ctx:
    battr = BoolAttr(Attribute.parse("true"))
    # CHECK: iattr value: True
    print("iattr value:", battr.value)

    # Test factory methods.
    # CHECK: default_get: true
    print("default_get:", BoolAttr.get(True))


# CHECK-LABEL: TEST: testFlatSymbolRefAttr
@run
def testFlatSymbolRefAttr():
  with Context() as ctx:
    sattr = FlatSymbolRefAttr(Attribute.parse('@symbol'))
    # CHECK: symattr value: symbol
    print("symattr value:", sattr.value)

    # Test factory methods.
    # CHECK: default_get: @foobar
    print("default_get:", FlatSymbolRefAttr.get("foobar"))


# CHECK-LABEL: TEST: testStringAttr
@run
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


# CHECK-LABEL: TEST: testNamedAttr
@run
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


# CHECK-LABEL: TEST: testDenseIntAttr
@run
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


# CHECK-LABEL: TEST: testDenseIntAttrGetItem
@run
def testDenseIntAttrGetItem():
  def print_item(attr_asm):
    attr = DenseIntElementsAttr(Attribute.parse(attr_asm))
    dtype = ShapedType(attr.type).element_type
    try:
      item = attr[0]
      print(f"{dtype}:", item)
    except TypeError as e:
      print(f"{dtype}:", e)

  with Context():
    # CHECK: i1: 1
    print_item("dense<true> : tensor<i1>")
    # CHECK: i8: 123
    print_item("dense<123> : tensor<i8>")
    # CHECK: i16: 123
    print_item("dense<123> : tensor<i16>")
    # CHECK: i32: 123
    print_item("dense<123> : tensor<i32>")
    # CHECK: i64: 123
    print_item("dense<123> : tensor<i64>")
    # CHECK: ui8: 123
    print_item("dense<123> : tensor<ui8>")
    # CHECK: ui16: 123
    print_item("dense<123> : tensor<ui16>")
    # CHECK: ui32: 123
    print_item("dense<123> : tensor<ui32>")
    # CHECK: ui64: 123
    print_item("dense<123> : tensor<ui64>")
    # CHECK: si8: -123
    print_item("dense<-123> : tensor<si8>")
    # CHECK: si16: -123
    print_item("dense<-123> : tensor<si16>")
    # CHECK: si32: -123
    print_item("dense<-123> : tensor<si32>")
    # CHECK: si64: -123
    print_item("dense<-123> : tensor<si64>")

    # CHECK: i7: Unsupported integer type
    print_item("dense<123> : tensor<i7>")


# CHECK-LABEL: TEST: testDenseFPAttr
@run
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


# CHECK-LABEL: TEST: testDictAttr
@run
def testDictAttr():
  with Context():
    dict_attr = {
      'stringattr':  StringAttr.get('string'),
      'integerattr' : IntegerAttr.get(
        IntegerType.get_signless(32), 42)
    }

    a = DictAttr.get(dict_attr)

    # CHECK attr: {integerattr = 42 : i32, stringattr = "string"}
    print("attr:", a)

    assert len(a) == 2

    # CHECK: 42 : i32
    print(a['integerattr'])

    # CHECK: "string"
    print(a['stringattr'])

    # CHECK: True
    print('stringattr' in a)

    # CHECK: False
    print('not_in_dict' in a)

    # Check that exceptions are raised as expected.
    try:
      _ = a['does_not_exist']
    except KeyError:
      pass
    else:
      assert False, "Exception not produced"

    try:
      _ = a[42]
    except IndexError:
      pass
    else:
      assert False, "expected IndexError on accessing an out-of-bounds attribute"

    # CHECK "empty: {}"
    print("empty: ", DictAttr.get())


# CHECK-LABEL: TEST: testTypeAttr
@run
def testTypeAttr():
  with Context():
    raw = Attribute.parse("vector<4xf32>")
    # CHECK: attr: vector<4xf32>
    print("attr:", raw)
    type_attr = TypeAttr(raw)
    # CHECK: f32
    print(ShapedType(type_attr.value).element_type)


# CHECK-LABEL: TEST: testArrayAttr
@run
def testArrayAttr():
  with Context():
    raw = Attribute.parse("[42, true, vector<4xf32>]")
  # CHECK: attr: [42, true, vector<4xf32>]
  print("raw attr:", raw)
  # CHECK: - 42
  # CHECK: - true
  # CHECK: - vector<4xf32>
  for attr in ArrayAttr(raw):
    print("- ", attr)

  with Context():
    intAttr = Attribute.parse("42")
    vecAttr = Attribute.parse("vector<4xf32>")
    boolAttr = BoolAttr.get(True)
    raw = ArrayAttr.get([vecAttr, boolAttr, intAttr])
  # CHECK: attr: [vector<4xf32>, true, 42]
  print("raw attr:", raw)
  # CHECK: - vector<4xf32>
  # CHECK: - true
  # CHECK: - 42
  arr = ArrayAttr(raw)
  for attr in arr:
    print("- ", attr)
  # CHECK: attr[0]: vector<4xf32>
  print("attr[0]:", arr[0])
  # CHECK: attr[1]: true
  print("attr[1]:", arr[1])
  # CHECK: attr[2]: 42
  print("attr[2]:", arr[2])
  try:
    print("attr[3]:", arr[3])
  except IndexError as e:
    # CHECK: Error: ArrayAttribute index out of range
    print("Error: ", e)
  with Context():
    try:
      ArrayAttr.get([None])
    except RuntimeError as e:
      # CHECK: Error: Invalid attribute (None?) when attempting to create an ArrayAttribute
      print("Error: ", e)
    try:
      ArrayAttr.get([42])
    except RuntimeError as e:
      # CHECK: Error: Invalid attribute when attempting to create an ArrayAttribute
      print("Error: ", e)

  with Context():
    array = ArrayAttr.get([StringAttr.get("a"), StringAttr.get("b")])
    array = array + [StringAttr.get("c")]
    # CHECK: concat: ["a", "b", "c"]
    print("concat: ", array)
