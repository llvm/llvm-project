# RUN: %PYTHON %s | FileCheck %s

import gc
import io
import itertools
from mlir.ir import *


def run(f):
    print("\nTEST:", f.__name__)
    f()
    gc.collect()
    assert Context._get_live_count() == 0
    return f


# CHECK-LABEL: TEST: testSymbolTableInsert
@run
def testSymbolTableInsert():
    with Context() as ctx:
        ctx.allow_unregistered_dialects = True
        m1 = Module.parse(
            """
      func.func private @foo()
      func.func private @bar()"""
        )
        m2 = Module.parse(
            """
      func.func private @qux()
      func.func private @foo()
      "foo.bar"() : () -> ()"""
        )

        symbol_table = SymbolTable(m1.operation)

        # CHECK: func private @foo
        # CHECK: func private @bar
        assert "foo" in symbol_table
        print(symbol_table["foo"])
        assert "bar" in symbol_table
        bar = symbol_table["bar"]
        print(symbol_table["bar"])

        assert "qux" not in symbol_table

        del symbol_table["bar"]
        try:
            symbol_table.erase(symbol_table["bar"])
        except KeyError:
            pass
        else:
            assert False, "expected KeyError"

        # CHECK: module
        # CHECK:   func private @foo()
        print(m1)
        assert "bar" not in symbol_table

        try:
            print(bar)
        except RuntimeError as e:
            if "the operation has been invalidated" not in str(e):
                raise
        else:
            assert False, "expected RuntimeError due to invalidated operation"

        qux = m2.body.operations[0]
        m1.body.append(qux)
        symbol_table.insert(qux)
        assert "qux" in symbol_table

        # Check that insertion actually renames this symbol in the symbol table.
        foo2 = m2.body.operations[0]
        m1.body.append(foo2)
        updated_name = symbol_table.insert(foo2)
        assert foo2.name.value != "foo"
        assert foo2.name == updated_name
        assert isinstance(updated_name, StringAttr)

        # CHECK: module
        # CHECK:   func private @foo()
        # CHECK:   func private @qux()
        # CHECK:   func private @foo{{.*}}
        print(m1)

        try:
            symbol_table.insert(m2.body.operations[0])
        except ValueError as e:
            if "Expected operation to have a symbol name" not in str(e):
                raise
        else:
            assert False, "exepcted ValueError when adding a non-symbol"


# CHECK-LABEL: testSymbolTableRAUW
@run
def testSymbolTableRAUW():
    with Context() as ctx:
        m = Module.parse(
            """
      func.func private @foo() {
        call @bar() : () -> ()
        return
      }
      func.func private @bar()
      """
        )
        foo, bar = list(m.operation.regions[0].blocks[0].operations)[0:2]

        # Do renaming just within `foo`.
        SymbolTable.set_symbol_name(bar, "bam")
        SymbolTable.replace_all_symbol_uses("bar", "bam", foo)
        # CHECK: call @bam()
        # CHECK: func private @bam
        print(m)
        # CHECK: Foo symbol: StringAttr("foo")
        # CHECK: Bar symbol: StringAttr("bam")
        print(f"Foo symbol: {repr(SymbolTable.get_symbol_name(foo))}")
        print(f"Bar symbol: {repr(SymbolTable.get_symbol_name(bar))}")

        # Do renaming within the module.
        SymbolTable.set_symbol_name(bar, "baz")
        SymbolTable.replace_all_symbol_uses("bam", "baz", m.operation)
        # CHECK: call @baz()
        # CHECK: func private @baz
        print(m)
        # CHECK: Foo symbol: StringAttr("foo")
        # CHECK: Bar symbol: StringAttr("baz")
        print(f"Foo symbol: {repr(SymbolTable.get_symbol_name(foo))}")
        print(f"Bar symbol: {repr(SymbolTable.get_symbol_name(bar))}")


# CHECK-LABEL: testSymbolTableVisibility
@run
def testSymbolTableVisibility():
    with Context() as ctx:
        m = Module.parse(
            """
      func.func private @foo() {
        return
      }
      """
        )
        foo = m.operation.regions[0].blocks[0].operations[0]
        # CHECK: Existing visibility: StringAttr("private")
        print(f"Existing visibility: {repr(SymbolTable.get_visibility(foo))}")
        SymbolTable.set_visibility(foo, "public")
        # CHECK: func public @foo
        print(m)


# CHECK: testWalkSymbolTables
@run
def testWalkSymbolTables():
    with Context() as ctx:
        m = Module.parse(
            """
      module @outer {
        module @inner{
        }
      }
      """
        )

        def callback(symbol_table_op, uses_visible):
            print(f"SYMBOL TABLE: {uses_visible}: {symbol_table_op}")

        # CHECK: SYMBOL TABLE: True: module @inner
        # CHECK: SYMBOL TABLE: True: module @outer
        SymbolTable.walk_symbol_tables(m.operation, True, callback)

        # Make sure exceptions in the callback are handled.
        def error_callback(symbol_table_op, uses_visible):
            assert False, "Raised from python"

        try:
            SymbolTable.walk_symbol_tables(m.operation, True, error_callback)
        except RuntimeError as e:
            # CHECK: GOT EXCEPTION: Exception raised in callback:
            # CHECK: AssertionError: Raised from python
            print(f"GOT EXCEPTION: {e}")
