from lldbsuite.test import lldbinline
from lldbsuite.test import decorators

decorators = [decorators.skipUnlessHasCallSiteInfo,
         decorators.skipIf(dwarf_version=['<', '4'])]
lldbinline.MakeInlineTest(__file__, globals(),
        name="DisambiguatePathsToCommonSink_V5",
        build_dict=dict(CFLAGS_EXTRAS="-O2 -glldb"), decorators=decorators)
lldbinline.MakeInlineTest(__file__, globals(),
        name="DisambiguatePathsToCommonSink_GNU",
        build_dict=dict(CFLAGS_EXTRAS="-O2 -ggdb"), decorators=decorators)
