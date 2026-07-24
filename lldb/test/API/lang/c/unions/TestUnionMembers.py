import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    def test(self):
        self.build_and_run()

        # Reinterpreting a larger member through smaller ones depends on the
        # host byte order. The values below assume little-endian; on big-endian
        # hosts those assertions are skipped.
        # TODO: Add test assertions for big endian.
        little_endian = self.target().GetByteOrder() == lldb.eByteOrderLittle

        # Type punning: a float and an integer sharing the same 4 bytes. Both
        # members span the whole storage, so this is endian-independent.
        self.expect_var_path("fb", type="FloatBits")
        self.expect_var_path("fb.f", type="float", value="1")
        self.expect_var_path("fb.bits", type="unsigned int", value="1065353216")
        self.expect_expr(
            "fb.bits", result_type="unsigned int", result_value="1065353216"
        )

        # Basic aliasing: the same int seen as two shorts and as a single byte.
        self.expect_var_path("basic", type="Basic")
        self.expect_var_path("basic.n", type="int", value="287454020")
        self.expect_var_path("basic.halves", type="unsigned short[2]")
        self.expect_expr("basic.n", result_type="int", result_value="287454020")
        if little_endian:
            self.expect_var_path(
                "basic.halves[0]", type="unsigned short", value="13124"
            )
            self.expect_var_path("basic.halves[1]", type="unsigned short", value="4386")
            self.expect_var_path("basic.byte", type="unsigned char", value="'D'")

        # A union whose storage is also described by a struct.
        self.expect_var_path("ws", type="WithStruct")
        self.expect_var_path("ws.packed", type="int", value="131073")
        self.expect_var_path("ws.view", type="Halves")
        if little_endian:
            self.expect_var_path("ws.view.lo", type="short", value="1")
            self.expect_var_path("ws.view.hi", type="short", value="2")

        # A union nested inside another union. Both inner members sit at offset
        # 0, so they alias each other regardless of byte order.
        self.expect_var_path("nested", type="Nested")
        self.expect_var_path("nested.all", type="int", value="393221")
        self.assertEqual(
            self.frame().GetValueForVariablePath("nested.inner.a").GetValueAsSigned(),
            self.frame().GetValueForVariablePath("nested.inner.b").GetValueAsSigned(),
        )
        if little_endian:
            self.expect_var_path("nested.inner.a", type="short", value="5")
            self.expect_var_path("nested.inner.b", type="short", value="5")

        # Members of an anonymous union are reached directly on the struct.
        self.expect_var_path("anon", type="WithAnonUnion")
        self.expect_var_path("anon.tag", type="int", value="42")
        self.expect_var_path("anon.as_int", type="int", value="67305985")
        self.expect_var_path("anon.as_bytes", type="unsigned char[4]")
        if little_endian:
            for index, byte in enumerate([1, 2, 3, 4]):
                self.expect_var_path(
                    f"anon.as_bytes[{index}]",
                    type="unsigned char",
                    value=f"'\\x0{byte}'",
                )
