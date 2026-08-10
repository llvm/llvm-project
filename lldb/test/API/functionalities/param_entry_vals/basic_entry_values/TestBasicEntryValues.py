from lldbsuite.test import lldbinline
from lldbsuite.test.decorators import *
from lldbsuite.test import lldbplatformutil

supported_archs = ["x86_64", "aarch64", "arm64", "arm64e"]
decorators = [
    skipIf(archs=no_match(supported_archs)),
    skipIf(compiler="clang", compiler_version=["<", "11.0"]),
    skipUnlessHasCallSiteInfo,
    skipIf(dwarf_version=["<", "4"]),
]

lldbinline.MakeInlineTest(
    __file__,
    globals(),
    decorators=decorators,
    name="BasicEntryValues_V5",
    build_dict=dict(CXXFLAGS_EXTRAS="-O2 -glldb"),
)

lldbinline.MakeInlineTest(
    __file__,
    globals(),
    decorators=decorators + [skipIf(debug_info="dsym")],
    name="BasicEntryValues_GNU",
    build_dict=dict(CXXFLAGS_EXTRAS="-O2 -ggdb -gdwarf-4"),
)
