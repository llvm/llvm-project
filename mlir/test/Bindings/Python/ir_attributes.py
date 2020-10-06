# RUN: %PYTHON %s | FileCheck %s

import gc
import mlir

def run(f):
  print("\nTEST:", f.__name__)
  f()
  gc.collect()
  assert mlir.ir.Context._get_live_count() == 0


# CHECK-LABEL: TEST: testParsePrint
def testParsePrint():
  ctx = mlir.ir.Context()
  t = ctx.parse_attr('"hello"')
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
  ctx = mlir.ir.Context()
  try:
    t = ctx.parse_attr("BAD_ATTR_DOES_NOT_EXIST")
  except ValueError as e:
    # CHECK: Unable to parse attribute: 'BAD_ATTR_DOES_NOT_EXIST'
    print("testParseError:", e)
  else:
    print("Exception not produced")

run(testParseError)


# CHECK-LABEL: TEST: testAttrEq
def testAttrEq():
  ctx = mlir.ir.Context()
  a1 = ctx.parse_attr('"attr1"')
  a2 = ctx.parse_attr('"attr2"')
  a3 = ctx.parse_attr('"attr1"')
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
  ctx = mlir.ir.Context()
  a1 = ctx.parse_attr('"attr1"')
  not_an_attr = "foo"
  # CHECK: False
  print(a1 == not_an_attr)
  # CHECK: False
  print(a1 == None)
  # CHECK: True
  print(a1 != None)

run(testAttrEqDoesNotRaise)


# CHECK-LABEL: TEST: testStandardAttrCasts
def testStandardAttrCasts():
  ctx = mlir.ir.Context()
  a1 = ctx.parse_attr('"attr1"')
  astr = mlir.ir.StringAttr(a1)
  aself = mlir.ir.StringAttr(astr)
  # CHECK: Attribute("attr1")
  print(repr(astr))
  try:
    tillegal = mlir.ir.StringAttr(ctx.parse_attr("1.0"))
  except ValueError as e:
    # CHECK: ValueError: Cannot cast attribute to StringAttr (from Attribute(1.000000e+00 : f64))
    print("ValueError:", e)
  else:
    print("Exception not produced")

run(testStandardAttrCasts)


# CHECK-LABEL: TEST: testFloatAttr
def testFloatAttr():
  ctx = mlir.ir.Context()
  fattr = mlir.ir.FloatAttr(ctx.parse_attr("42.0 : f32"))
  # CHECK: fattr value: 42.0
  print("fattr value:", fattr.value)

  # Test factory methods.
  loc = ctx.get_unknown_location()
  # CHECK: default_get: 4.200000e+01 : f32
  print("default_get:", mlir.ir.FloatAttr.get(
      mlir.ir.F32Type(ctx), 42.0, loc))
  # CHECK: f32_get: 4.200000e+01 : f32
  print("f32_get:", mlir.ir.FloatAttr.get_f32(ctx, 42.0))
  # CHECK: f64_get: 4.200000e+01 : f64
  print("f64_get:", mlir.ir.FloatAttr.get_f64(ctx, 42.0))
  try:
    fattr_invalid = mlir.ir.FloatAttr.get(
        mlir.ir.IntegerType.get_signless(ctx, 32), 42, loc)
  except ValueError as e:
    # CHECK: invalid 'Type(i32)' and expected floating point type.
    print(e)
  else:
    print("Exception not produced")

run(testFloatAttr)


# CHECK-LABEL: TEST: testIntegerAttr
def testIntegerAttr():
  ctx = mlir.ir.Context()
  iattr = mlir.ir.IntegerAttr(ctx.parse_attr("42"))
  # CHECK: iattr value: 42
  print("iattr value:", iattr.value)

  # Test factory methods.
  # CHECK: default_get: 42 : i32
  print("default_get:", mlir.ir.IntegerAttr.get(
      mlir.ir.IntegerType.get_signless(ctx, 32), 42))

run(testIntegerAttr)


# CHECK-LABEL: TEST: testBoolAttr
def testBoolAttr():
  ctx = mlir.ir.Context()
  battr = mlir.ir.BoolAttr(ctx.parse_attr("true"))
  # CHECK: iattr value: 1
  print("iattr value:", battr.value)

  # Test factory methods.
  # CHECK: default_get: true
  print("default_get:", mlir.ir.BoolAttr.get(ctx, True))

run(testBoolAttr)


# CHECK-LABEL: TEST: testStringAttr
def testStringAttr():
  ctx = mlir.ir.Context()
  sattr = mlir.ir.StringAttr(ctx.parse_attr('"stringattr"'))
  # CHECK: sattr value: stringattr
  print("sattr value:", sattr.value)

  # Test factory methods.
  # CHECK: default_get: "foobar"
  print("default_get:", mlir.ir.StringAttr.get(ctx, "foobar"))
  # CHECK: typed_get: "12345" : i32
  print("typed_get:", mlir.ir.StringAttr.get_typed(
      mlir.ir.IntegerType.get_signless(ctx, 32), "12345"))

run(testStringAttr)


# CHECK-LABEL: TEST: testNamedAttr
def testNamedAttr():
  ctx = mlir.ir.Context()
  a = ctx.parse_attr('"stringattr"')
  named = a.get_named("foobar")  # Note: under the small object threshold
  # CHECK: attr: "stringattr"
  print("attr:", named.attr)
  # CHECK: name: foobar
  print("name:", named.name)
  # CHECK: named: NamedAttribute(foobar="stringattr")
  print("named:", named)

run(testNamedAttr)
