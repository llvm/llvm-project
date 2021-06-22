from lldbsuite.test import lldbinline
from lldbsuite.test import decorators

decor = [decorators.skipUnlessHasCallSiteInfo,
         decorators.skipIf(dwarf_version=['<', '4']),
         decorators.skipIf(compiler="clang", compiler_version=['<', '11.0'])]
lldbinline.MakeInlineTest(__file__, globals(), name="InliningAndTailCalls_V5",
        build_dict=dict(CFLAGS_EXTRAS="-O2 -glldb"), decorators=decor)
lldbinline.MakeInlineTest(__file__, globals(), name="InliningAndTailCalls_GNU",
        build_dict=dict(CFLAGS_EXTRAS="-O2 -ggdb"),
        decorators=decor+[decorators.skipIf(debug_info="dsym")])
