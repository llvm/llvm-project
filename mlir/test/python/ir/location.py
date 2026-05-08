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
        loc = UnknownLoc.get()
    assert loc.context is ctx
    ctx = None
    gc.collect()
    # CHECK: unknown str: loc(unknown)
    print("unknown str:", str(loc))
    # CHECK: unknown repr: UnknownLoc(loc(unknown))
    print("unknown repr:", repr(loc))

    assert isinstance(loc, UnknownLoc)
    assert isinstance(loc, Location)
    assert not isinstance(loc, FileLineColLoc)
    assert not isinstance(loc, NameLoc)
    assert not isinstance(loc, CallSiteLoc)
    assert not isinstance(loc, FusedLoc)


run(testUnknown)


# CHECK-LABEL: TEST: testLocationAttr
def testLocationAttr():
    with Context() as ctxt:
        loc = UnknownLoc.get()
        attr = loc.attr
        clone = Location.from_attr(attr)
    gc.collect()
    # CHECK: loc: loc(unknown)
    print("loc:", str(loc))
    # CHECK: clone: loc(unknown)
    print("clone:", str(clone))
    assert loc == clone
    assert isinstance(clone, UnknownLoc)


run(testLocationAttr)


# CHECK-LABEL: TEST: testFileLineCol
def testFileLineCol():
    with Context() as ctx:
        loc = FileLineColLoc.get("foo1.txt", 123, 56)
        range = FileLineColLoc.get("foo2.txt", 123, 56, 124, 100)

    ctx = None
    gc.collect()

    # CHECK: file str: loc("foo1.txt":123:56)
    print("file str:", str(loc))
    # CHECK: file repr: FileLineColLoc(loc("foo1.txt":123:56))
    print("file repr:", repr(loc))
    # CHECK: file range str: loc("foo2.txt":123:56 to 124:100)
    print("file range str:", str(range))
    # CHECK: file range repr: FileLineColLoc(loc("foo2.txt":123:56 to 124:100))
    print("file range repr:", repr(range))

    assert isinstance(loc, FileLineColLoc)
    assert not isinstance(loc, NameLoc)
    assert not isinstance(loc, CallSiteLoc)
    assert not isinstance(loc, FusedLoc)

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

    assert isinstance(range, FileLineColLoc)
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
        loc = FileLineColLoc.get("foo3.txt", 127, 61)
        with loc:
            i32 = IntegerType.get_signless(32)
            module = Module.create()
            with InsertionPoint(module.body):
                op = Operation.create("custom.op1", results=[i32])
                new_value = op.result
                # CHECK: new_value location: loc("foo3.txt":127:61)
                print("new_value location: ", new_value.location)
                # `op.location` and `value.location` both downcast to the
                # concrete subclass.
                assert isinstance(op.location, FileLineColLoc)
                assert isinstance(new_value.location, FileLineColLoc)
                assert op.location.typeid == FileLineColLoc.static_typeid


run(testFileLineCol)


# CHECK-LABEL: TEST: testName
def testName():
    with Context() as ctx:
        loc = NameLoc.get("nombre")
        loc_with_child_loc = NameLoc.get("naam", loc)

    ctx = None
    gc.collect()

    # CHECK: name str: loc("nombre")
    print("name str:", str(loc))
    # CHECK: name repr: NameLoc(loc("nombre"))
    print("name repr:", repr(loc))
    # CHECK: name str: loc("naam"("nombre"))
    print("name str:", str(loc_with_child_loc))
    # CHECK: name repr: NameLoc(loc("naam"("nombre")))
    print("name repr:", repr(loc_with_child_loc))

    assert isinstance(loc, NameLoc)
    # CHECK: name name_str: nombre
    print("name name_str:", loc.name_str)
    # CHECK: name child_loc: loc(unknown)
    print("name child_loc:", loc.child_loc)
    assert isinstance(loc.child_loc, UnknownLoc)

    assert isinstance(loc_with_child_loc, NameLoc)
    # CHECK: name name_str: naam
    print("name name_str:", loc_with_child_loc.name_str)
    # CHECK: name child_loc_with_child_loc: loc("nombre")
    print("name child_loc_with_child_loc:", loc_with_child_loc.child_loc)
    assert isinstance(loc_with_child_loc.child_loc, NameLoc)


run(testName)


# CHECK-LABEL: TEST: testCallSite
def testCallSite():
    with Context() as ctx:
        loc = CallSiteLoc.get(
            FileLineColLoc.get("foo.text", 123, 45),
            [
                FileLineColLoc.get("util.foo", 379, 21),
                FileLineColLoc.get("main.foo", 100, 63),
            ],
        )
    ctx = None
    # CHECK: callsite str: loc(callsite("foo.text":123:45 at callsite("util.foo":379:21 at "main.foo":100:63))
    print("callsite str:", str(loc))
    # CHECK: callsite repr: CallSiteLoc(loc(callsite("foo.text":123:45 at callsite("util.foo":379:21 at "main.foo":100:63)))
    print("callsite repr:", repr(loc))

    assert isinstance(loc, CallSiteLoc)
    # CHECK: callsite callee: loc("foo.text":123:45)
    print("callsite callee:", loc.callee)
    assert isinstance(loc.callee, FileLineColLoc)
    # CHECK: callsite caller: loc(callsite("util.foo":379:21 at "main.foo":100:63))
    print("callsite caller:", loc.caller)
    assert isinstance(loc.caller, CallSiteLoc)


run(testCallSite)


# CHECK-LABEL: TEST: testFused
def testFused():
    with Context() as ctx:
        loc_single = Location.fused([NameLoc.get("apple")])
        loc_empty = Location.fused([])
        loc = FusedLoc.get([NameLoc.get("apple"), NameLoc.get("banana")])
        attr = Attribute.parse('"sauteed"')
        loc_attr = FusedLoc.get([NameLoc.get("carrot"), NameLoc.get("potatoes")], attr)
        loc_empty_attr = FusedLoc.get([], attr)
        loc_single_attr = FusedLoc.get([NameLoc.get("apple")], attr)

        try:
            FusedLoc.get([NameLoc.get("x")])
        except ValueError as e:
            # CHECK: fused strict error: FusedLoc.get would collapse
            print("fused strict error:", str(e)[:35])
        else:
            assert False, "expected ValueError from strict FusedLoc.get"

    ctx = None

    assert not isinstance(loc_single, FusedLoc)
    assert isinstance(loc_single, NameLoc)
    # CHECK: fused str: loc("apple")
    print("fused str:", str(loc_single))
    # CHECK: fused repr: NameLoc(loc("apple"))
    print("fused repr:", repr(loc_single))

    assert isinstance(loc, FusedLoc)
    # CHECK: fused str: loc(fused["apple", "banana"])
    print("fused str:", str(loc))
    # CHECK: fused repr: FusedLoc(loc(fused["apple", "banana"]))
    print("fused repr:", repr(loc))
    # CHECK: fused locations: [NameLoc(loc("apple")), NameLoc(loc("banana"))]
    print("fused locations:", loc.locations)
    # CHECK: fused metadata: None
    print("fused metadata:", loc.metadata)

    assert isinstance(loc_attr, FusedLoc)
    # CHECK: fused metadata: "sauteed"
    print("fused metadata:", loc_attr.metadata)
    # CHECK: fused str: loc(fused<"sauteed">["carrot", "potatoes"])
    print("fused str:", str(loc_attr))
    # CHECK: fused repr: FusedLoc(loc(fused<"sauteed">["carrot", "potatoes"]))
    print("fused repr:", repr(loc_attr))
    # CHECK: fused locations: [NameLoc(loc("carrot")), NameLoc(loc("potatoes"))]
    print("fused locations:", loc_attr.locations)

    assert not isinstance(loc_empty, FusedLoc)
    assert isinstance(loc_empty, UnknownLoc)
    # CHECK: fused str: loc(unknown)
    print("fused str:", str(loc_empty))
    # CHECK: fused repr: UnknownLoc(loc(unknown))
    print("fused repr:", repr(loc_empty))

    assert isinstance(loc_empty_attr, FusedLoc)
    # CHECK: fused str: loc(fused<"sauteed">[unknown])
    print("fused str:", str(loc_empty_attr))
    # CHECK: fused repr: FusedLoc(loc(fused<"sauteed">[unknown]))
    print("fused repr:", repr(loc_empty_attr))
    # CHECK: fused locations: [UnknownLoc(loc(unknown))]
    print("fused locations:", loc_empty_attr.locations)

    assert isinstance(loc_single_attr, FusedLoc)
    # CHECK: fused str: loc(fused<"sauteed">["apple"])
    print("fused str:", str(loc_single_attr))
    # CHECK: fused repr: FusedLoc(loc(fused<"sauteed">["apple"]))
    print("fused repr:", repr(loc_single_attr))
    # CHECK: fused locations: [NameLoc(loc("apple"))]
    print("fused locations:", loc_single_attr.locations)


run(testFused)


# CHECK-LABEL: TEST: testCast
def testCast():
    with Context() as ctx:
        unknown = UnknownLoc.get()
        as_unknown = UnknownLoc(unknown)
        assert isinstance(as_unknown, UnknownLoc)

        try:
            FileLineColLoc(unknown)
        except ValueError as e:
            # CHECK: cast error: Cannot cast location to FileLineColLoc (from loc(unknown))
            print("cast error:", str(e))
        else:
            assert False, "expected ValueError"

    ctx = None


run(testCast)


# CHECK-LABEL: TEST: testLocationCapsule
def testLocationCapsule():
    with Context() as ctx:
        loc1 = FileLineColLoc.get("foo.txt", 123, 56)
    # CHECK: mlir.ir.Location._CAPIPtr
    loc_capsule = loc1._CAPIPtr
    print(loc_capsule)
    loc2 = Location._CAPICreate(loc_capsule)
    assert loc2 == loc1
    assert loc2.context is ctx


run(testLocationCapsule)
