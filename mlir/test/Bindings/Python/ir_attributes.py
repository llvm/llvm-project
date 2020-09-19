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
