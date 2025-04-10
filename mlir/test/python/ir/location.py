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
        loc = Location.file("foo1.txt", 123, 56)
        range = Location.file("foo2.txt", 123, 56, 124, 100)

    ctx = None
    gc.collect()

    # CHECK: file str: loc("foo1.txt":123:56)
    print("file str:", str(loc))
    # CHECK: file repr: loc("foo1.txt":123:56)
    print("file repr:", repr(loc))
    # CHECK: file range str: loc("foo2.txt":123:56 to 124:100)
    print("file range str:", str(range))
    # CHECK: file range repr: loc("foo2.txt":123:56 to 124:100)
    print("file range repr:", repr(range))

    assert loc.is_a_file()
    assert not loc.is_a_name()
    assert not loc.is_a_callsite()
    assert not loc.is_a_fused()

    # CHECK: file filename: foo1.txt
    print("file filename:", loc.filename)
    # CHECK: file start_line: 123
    print("file start_line:", loc.start_line)
    # CHECK: file start_col: 56
    print("file start_col:", loc.start_col)
    # CHECK: file end_line: 123
    print("file end_line:", loc.end_line)
    # CHECK: file end_col: 56
    print("file end_col:", loc.end_col)

    assert range.is_a_file()
    # CHECK: file filename: foo2.txt
    print("file filename:", range.filename)
    # CHECK: file start_line: 123
    print("file start_line:", range.start_line)
    # CHECK: file start_col: 56
    print("file start_col:", range.start_col)
    # CHECK: file end_line: 124
    print("file end_line:", range.end_line)
    # CHECK: file end_col: 100
    print("file end_col:", range.end_col)

    with Context() as ctx:
        ctx.allow_unregistered_dialects = True
        loc = Location.file("foo3.txt", 127, 61)
        with loc:
            i32 = IntegerType.get_signless(32)
            module = Module.create()
            with InsertionPoint(module.body):
                new_value = Operation.create("custom.op1", results=[i32]).result
                # CHECK: new_value location: loc("foo3.txt":127:61)
                print("new_value location: ", new_value.location)


run(testFileLineCol)


# CHECK-LABEL: TEST: testName
def testName():
    with Context() as ctx:
        loc = Location.name("nombre")
        loc_with_child_loc = Location.name("naam", loc)

    ctx = None
    gc.collect()

    # CHECK: name str: loc("nombre")
    print("name str:", str(loc))
    # CHECK: name repr: loc("nombre")
    print("name repr:", repr(loc))
    # CHECK: name str: loc("naam"("nombre"))
    print("name str:", str(loc_with_child_loc))
    # CHECK: name repr: loc("naam"("nombre"))
    print("name repr:", repr(loc_with_child_loc))

    assert loc.is_a_name()
    # CHECK: name name_str: nombre
    print("name name_str:", loc.name_str)
    # CHECK: name child_loc: loc(unknown)
    print("name child_loc:", loc.child_loc)

    assert loc_with_child_loc.is_a_name()
    # CHECK: name name_str: naam
    print("name name_str:", loc_with_child_loc.name_str)
    # CHECK: name child_loc_with_child_loc: loc("nombre")
    print("name child_loc_with_child_loc:", loc_with_child_loc.child_loc)


run(testName)


# CHECK-LABEL: TEST: testCallSite
def testCallSite():
    with Context() as ctx:
        loc = Location.callsite(
            Location.file("foo.text", 123, 45),
            [Location.file("util.foo", 379, 21), Location.file("main.foo", 100, 63)],
        )
    ctx = None
    # CHECK: callsite str: loc(callsite("foo.text":123:45 at callsite("util.foo":379:21 at "main.foo":100:63))
    print("callsite str:", str(loc))
    # CHECK: callsite repr: loc(callsite("foo.text":123:45 at callsite("util.foo":379:21 at "main.foo":100:63))
    print("callsite repr:", repr(loc))

    assert loc.is_a_callsite()

    # CHECK: callsite callee: loc("foo.text":123:45)
    print("callsite callee:", loc.callee)
    # CHECK: callsite caller: loc(callsite("util.foo":379:21 at "main.foo":100:63))
    print("callsite caller:", loc.caller)


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

    assert not loc_single.is_a_fused()
    # CHECK: fused str: loc("apple")
    print("fused str:", str(loc_single))
    # CHECK: fused repr: loc("apple")
    print("fused repr:", repr(loc_single))
    # # CHECK: fused locations: []
    print("fused locations:", loc_single.locations)

    assert loc.is_a_fused()
    # CHECK: fused str: loc(fused["apple", "banana"])
    print("fused str:", str(loc))
    # CHECK: fused repr: loc(fused["apple", "banana"])
    print("fused repr:", repr(loc))
    # CHECK: fused locations: [loc("apple"), loc("banana")]
    print("fused locations:", loc.locations)

    assert loc_attr.is_a_fused()
    # CHECK: fused str: loc(fused<"sauteed">["carrot", "potatoes"])
    print("fused str:", str(loc_attr))
    # CHECK: fused repr: loc(fused<"sauteed">["carrot", "potatoes"])
    print("fused repr:", repr(loc_attr))
    # CHECK: fused locations: [loc("carrot"), loc("potatoes")]
    print("fused locations:", loc_attr.locations)

    assert not loc_empty.is_a_fused()
    # CHECK: fused str: loc(unknown)
    print("fused str:", str(loc_empty))
    # CHECK: fused repr: loc(unknown)
    print("fused repr:", repr(loc_empty))
    # CHECK: fused locations: []
    print("fused locations:", loc_empty.locations)

    assert loc_empty_attr.is_a_fused()
    # CHECK: fused str: loc(fused<"sauteed">[unknown])
    print("fused str:", str(loc_empty_attr))
    # CHECK: fused repr: loc(fused<"sauteed">[unknown])
    print("fused repr:", repr(loc_empty_attr))
    # CHECK: fused locations: [loc(unknown)]
    print("fused locations:", loc_empty_attr.locations)

    assert loc_single_attr.is_a_fused()
    # CHECK: fused str: loc(fused<"sauteed">["apple"])
    print("fused str:", str(loc_single_attr))
    # CHECK: fused repr: loc(fused<"sauteed">["apple"])
    print("fused repr:", repr(loc_single_attr))
    # CHECK: fused locations: [loc("apple")]
    print("fused locations:", loc_single_attr.locations)


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
