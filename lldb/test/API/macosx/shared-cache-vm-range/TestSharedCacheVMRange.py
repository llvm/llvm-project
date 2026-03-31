"""
Test that we can get the vm range of the shared cache.
"""

import lldb
import re
import json
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class SharedCacheVMRangeTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipIfRemote
    @skipUnlessDarwin
    @skipIfOutOfTreeDebugserver  # debugserver returns shared_cache_size
    def test_shared_cache_vm_range(self):
        """Test that the shared cache VM range contains a known libc function"""
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.c")
        )

        res = lldb.SBCommandReturnObject()
        ci = self.dbg.GetCommandInterpreter()
        self.assertTrue(ci, VALID_COMMAND_INTERPRETER)

        ci.HandleCommand("process plugin packet send jGetSharedCacheInfo:{}", res)

        # packet: jGetSharedCacheInfo:{}
        # response:
        # {
        #   "shared_cache_base_address": 6572900352,
        #   "shared_cache_uuid": "674DB25A-34B2-3C56-8BD4-7D78005B2F2E",
        #   "no_shared_cache": false,
        #   "shared_cache_private_cache": false,
        #   "shared_cache_path": "/System/Volumes/Preboot/Cryptexes/OS/System/Library/dyld/dyld_shared_cache_arm64e",
        #   "shared_cache_size": 5820792832
        # }

        self.assertTrue("response: " in res.GetOutput())
        response = re.search("response: (.+)", res.GetOutput()).group(1)
        json_response = json.loads(response)
        self.assertTrue("shared_cache_base_address" in json_response)
        self.assertTrue("shared_cache_size" in json_response)
        start = json_response["shared_cache_base_address"]
        end = start + json_response["shared_cache_size"]

        symctx_list = target.FindSymbols("printf", lldb.eSymbolTypeCode)
        self.assertGreater(symctx_list.GetSize(), 0)

        symctx = symctx_list.GetContextAtIndex(0)
        sym = symctx.GetSymbol()
        self.assertTrue(sym.IsValid())
        addr = sym.GetStartAddress()
        printf_load_addr = addr.GetLoadAddress(target)

        self.assertGreater(printf_load_addr, start)
        self.assertLess(printf_load_addr, end)
