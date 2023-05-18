# RUN: %PYTHON %s | FileCheck %s

import gc
from mlir.ir import *


def run(f):
    print("\nTEST:", f.__name__)
    f()
    gc.collect()
    assert Context._get_live_count() == 0


# CHECK-LABEL: TEST: testUnknown
def testUnknown():
    with Context() as ctx:
        loc = Location.unknown()
    assert loc.context is ctx
    ctx = None
    gc.collect()
    # CHECK: unknown str: loc(unknown)
    print("unknown str:", str(loc))
    # CHECK: unknown repr: loc(unknown)
    print("unknown repr:", repr(loc))


run(testUnknown)


# CHECK-LABEL: TEST: testLocationAttr
def testLocationAttr():
    with Context() as ctxt:
        loc = Location.unknown()
        attr = loc.attr
        clone = Location.from_attr(attr)
    gc.collect()
    # CHECK: loc: loc(unknown)
    print("loc:", str(loc))
    # CHECK: clone: loc(unknown)
    print("clone:", str(clone))
    assert loc == clone


run(testLocationAttr)

# CHECK-LABEL: TEST: testFileLineCol
def testFileLineCol():
    with Context() as ctx:
        loc = Location.file("foo.txt", 123, 56)
    ctx = None
    gc.collect()
    # CHECK: file str: loc("foo.txt":123:56)
    print("file str:", str(loc))
    # CHECK: file repr: loc("foo.txt":123:56)
    print("file repr:", repr(loc))


run(testFileLineCol)


# CHECK-LABEL: TEST: testName
def testName():
    with Context() as ctx:
        loc = Location.name("nombre")
        locWithChildLoc = Location.name("naam", loc)
    ctx = None
    gc.collect()
    # CHECK: file str: loc("nombre")
    print("file str:", str(loc))
    # CHECK: file repr: loc("nombre")
    print("file repr:", repr(loc))
    # CHECK: file str: loc("naam"("nombre"))
    print("file str:", str(locWithChildLoc))
    # CHECK: file repr: loc("naam"("nombre"))
    print("file repr:", repr(locWithChildLoc))


run(testName)


# CHECK-LABEL: TEST: testCallSite
def testCallSite():
    with Context() as ctx:
        loc = Location.callsite(
            Location.file("foo.text", 123, 45),
            [Location.file("util.foo", 379, 21), Location.file("main.foo", 100, 63)],
        )
    ctx = None
    # CHECK: file str: loc(callsite("foo.text":123:45 at callsite("util.foo":379:21 at "main.foo":100:63))
    print("file str:", str(loc))
    # CHECK: file repr: loc(callsite("foo.text":123:45 at callsite("util.foo":379:21 at "main.foo":100:63))
    print("file repr:", repr(loc))


run(testCallSite)


# CHECK-LABEL: TEST: testFused
def testFused():
    with Context() as ctx:
        loc_single = Location.fused([Location.name("apple")])
        loc = Location.fused([Location.name("apple"), Location.name("banana")])
        attr = Attribute.parse('"sauteed"')
        loc_attr = Location.fused(
            [Location.name("carrot"), Location.name("potatoes")], attr
        )
        loc_empty = Location.fused([])
        loc_empty_attr = Location.fused([], attr)
        loc_single_attr = Location.fused([Location.name("apple")], attr)
    ctx = None
    # CHECK: file str: loc("apple")
    print("file str:", str(loc_single))
    # CHECK: file repr: loc("apple")
    print("file repr:", repr(loc_single))
    # CHECK: file str: loc(fused["apple", "banana"])
    print("file str:", str(loc))
    # CHECK: file repr: loc(fused["apple", "banana"])
    print("file repr:", repr(loc))
    # CHECK: file str: loc(fused<"sauteed">["carrot", "potatoes"])
    print("file str:", str(loc_attr))
    # CHECK: file repr: loc(fused<"sauteed">["carrot", "potatoes"])
    print("file repr:", repr(loc_attr))
    # CHECK: file str: loc(unknown)
    print("file str:", str(loc_empty))
    # CHECK: file repr: loc(unknown)
    print("file repr:", repr(loc_empty))
    # CHECK: file str: loc(fused<"sauteed">[unknown])
    print("file str:", str(loc_empty_attr))
    # CHECK: file repr: loc(fused<"sauteed">[unknown])
    print("file repr:", repr(loc_empty_attr))
    # CHECK: file str: loc(fused<"sauteed">["apple"])
    print("file str:", str(loc_single_attr))
    # CHECK: file repr: loc(fused<"sauteed">["apple"])
    print("file repr:", repr(loc_single_attr))


run(testFused)


# CHECK-LABEL: TEST: testLocationCapsule
def testLocationCapsule():
    with Context() as ctx:
        loc1 = Location.file("foo.txt", 123, 56)
    # CHECK: mlir.ir.Location._CAPIPtr
    loc_capsule = loc1._CAPIPtr
    print(loc_capsule)
    loc2 = Location._CAPICreate(loc_capsule)
    assert loc2 == loc1
    assert loc2.context is ctx


run(testLocationCapsule)
