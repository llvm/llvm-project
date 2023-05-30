"""
Test lldb breakpoint ids.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCPPBreakpointLocations(TestBase):
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24764")
    def test(self):
        self.do_test(dict())

    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24764")
    @skipIf(compiler=no_match("clang"))
    @skipIf(compiler_version=["<", "15.0"])
    def test_simple_template_names(self):
        self.do_test(dict(CFLAGS_EXTRAS="-gsimple-template-names"))

    def do_test(self, debug_flags):
        self.build(dictionary=debug_flags)
        self.breakpoint_id_tests()

    def verify_breakpoint_locations(self, target, bp_dict):
        name = bp_dict["name"]
        names = bp_dict["loc_names"]
        bp = target.BreakpointCreateByName(name)
        self.assertEquals(
            bp.GetNumLocations(),
            len(names),
            "Make sure we find the right number of breakpoint locations for {}".format(
                name
            ),
        )

        bp_loc_names = list()
        for bp_loc in bp:
            bp_loc_names.append(bp_loc.GetAddress().GetFunction().GetName())

        for name in names:
            found = name in bp_loc_names
            if not found:
                print("Didn't find '%s' in: %s" % (name, bp_loc_names))
            self.assertTrue(found, "Make sure we find all required locations")

    def breakpoint_id_tests(self):
        # Create a target by the debugger.
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)
        bp_dicts = [
            {
                "name": "func1",
                "loc_names": ["a::c::func1()", "aa::cc::func1()", "b::c::func1()"],
            },
            {
                "name": "func2",
                "loc_names": ["a::c::func2()", "aa::cc::func2()", "c::d::func2()"],
            },
            {
                "name": "func3",
                "loc_names": [
                    "a::c::func3()",
                    "aa::cc::func3()",
                    "b::c::func3()",
                    "c::d::func3()",
                ],
            },
            {"name": "c::func1", "loc_names": ["a::c::func1()", "b::c::func1()"]},
            {"name": "c::func2", "loc_names": ["a::c::func2()"]},
            {"name": "c::func3", "loc_names": ["a::c::func3()", "b::c::func3()"]},
            {"name": "a::c::func1", "loc_names": ["a::c::func1()"]},
            {"name": "b::c::func1", "loc_names": ["b::c::func1()"]},
            {"name": "c::d::func2", "loc_names": ["c::d::func2()"]},
            {"name": "a::c::func1()", "loc_names": ["a::c::func1()"]},
            {"name": "b::c::func1()", "loc_names": ["b::c::func1()"]},
            {"name": "c::d::func2()", "loc_names": ["c::d::func2()"]},
            # Template cases
            {"name": "func<float>", "loc_names": []},
            {"name": "Foo::func<float>", "loc_names": []},
            {"name": "ns::Foo::func<float>", "loc_names": []},
            {"name": "func<int>", "loc_names": ["auto ns::Foo<double>::func<int>()"]},
            {
                "name": "Foo<double>::func<int>",
                "loc_names": ["auto ns::Foo<double>::func<int>()"],
            },
            {
                "name": "ns::Foo<double>::func<int>",
                "loc_names": ["auto ns::Foo<double>::func<int>()"],
            },
            {
                "name": "func",
                "loc_names": [
                    "auto ns::Foo<double>::func<int>()",
                    "auto ns::Foo<double>::func<ns::Foo<int>>()",
                ],
            },
            {"name": "operator", "loc_names": []},
            {
                "name": "ns::Foo<double>::operator bool",
                "loc_names": ["ns::Foo<double>::operator bool()"],
            },
            {
                "name": "operator a::c",
                "loc_names": ["ns::Foo<double>::operator a::c<a::c>()"],
            },
            {
                "name": "operator ns::Foo<int>",
                "loc_names": ["ns::Foo<double>::operator ns::Foo<int><ns::Foo<int>>()"],
            },
            {"name": "operator<<<a::c>", "loc_names": []},
            {
                "name": "operator<<<int>",
                "loc_names": ["void ns::Foo<double>::operator<<<int>(int)"],
            },
            {
                "name": "ns::Foo<double>::operator<<",
                "loc_names": [
                    "void ns::Foo<double>::operator<<<int>(int)",
                    "void ns::Foo<double>::operator<<<ns::Foo<int>>(ns::Foo<int>)",
                ],
            },
            {"name": "g<float>", "loc_names": []},
            {"name": "g<int>", "loc_names": ["void ns::g<int>()"]},
            {"name": "g<char>", "loc_names": ["void ns::g<char>()"]},
            {"name": "g", "loc_names": ["void ns::g<int>()", "void ns::g<char>()"]},
            {"name": "ns::g<float>", "loc_names": []},
            {"name": "ns::g<int>", "loc_names": ["void ns::g<int>()"]},
            {"name": "ns::g<char>", "loc_names": ["void ns::g<char>()"]},
            {"name": "ns::g", "loc_names": ["void ns::g<int>()", "void ns::g<char>()"]},
        ]

        for bp_dict in bp_dicts:
            self.verify_breakpoint_locations(target, bp_dict)

    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24764")
    def test_destructors(self):
        self.build()
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)

        # Don't skip prologue, so we can check the breakpoint address more
        # easily
        self.runCmd("settings set target.skip-prologue false")
        try:
            names = ["~c", "c::~c", "c::~c()"]
            loc_names = {"a::c::~c()", "b::c::~c()"}
            # TODO: For windows targets we should put windows mangled names
            # here
            symbols = ["_ZN1a1cD1Ev", "_ZN1a1cD2Ev", "_ZN1b1cD1Ev", "_ZN1b1cD2Ev"]

            for name in names:
                bp = target.BreakpointCreateByName(name)

                bp_loc_names = {
                    bp_loc.GetAddress().GetFunction().GetName() for bp_loc in bp
                }
                self.assertEquals(
                    bp_loc_names, loc_names, "Breakpoint set on the correct symbol"
                )

                bp_addresses = {bp_loc.GetLoadAddress() for bp_loc in bp}
                symbol_addresses = set()
                for symbol in symbols:
                    sc_list = target.FindSymbols(symbol, lldb.eSymbolTypeCode)
                    self.assertEquals(sc_list.GetSize(), 1, "Found symbol " + symbol)
                    symbol = sc_list.GetContextAtIndex(0).GetSymbol()
                    symbol_addresses.add(
                        symbol.GetStartAddress().GetLoadAddress(target)
                    )

                self.assertEquals(
                    symbol_addresses, bp_addresses, "Breakpoint set on correct address"
                )
        finally:
            self.runCmd("settings clear target.skip-prologue")
