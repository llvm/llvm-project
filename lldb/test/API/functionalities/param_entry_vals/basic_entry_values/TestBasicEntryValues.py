from lldbsuite.test import lldbinline
from lldbsuite.test.decorators import *
from lldbsuite.test import lldbplatformutil

supported_archs = ["x86_64", "aarch64"]
decorators = [skipIf(archs=no_match(supported_archs)),
         skipIf(compiler="clang", compiler_version=['<', '10.0']),
         skipUnlessHasCallSiteInfo,
         skipIf(dwarf_version=['<', '4'])]

lldbinline.MakeInlineTest(__file__, globals(), decorators=decorators,
        name="BasicEntryValues_V5",
        build_dict=dict(CXXFLAGS_EXTRAS="-O2 -glldb"))
