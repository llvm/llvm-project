"""
Make sure the plugin list, enable, and disable commands work.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class TestFrameVar(TestBase):
    # If your test case doesn't stress debug info, then
    # set this to true.  That way it won't be run once for
    # each debug info format.
    NO_DEBUG_INFO_TESTCASE = True

    def test_plugin_list_enable_disable_commands(self):
        for plugin_namespace in [
            "abi",
            "architecture",
            "disassembler",
            "dynamic-loader",
            "emulate-instruction",
            "instrumentation-runtime",
            "jit-loader",
            "language",
            "language-runtime",
            "memory-history",
            "object-container",
            "object-file",
            "operating-system",
            "platform",
            "process",
            "repl",
            "register-type-builder",
            "script-interpreter",
            "scripted-interface",
            "structured-data",
            "symbol-file",
            "symbol-locator",
            "symbol-vendor",
            "system-runtime",
            # 'trace', # No trace plugin is registered by default.
            "trace-exporter",
            "type-system",
            "unwind-assembly",
        ]:
            self.do_list_disable_enable_test(plugin_namespace)

    def do_list_disable_enable_test(self, plugin_namespace):
        # Plugins are enabled by default.
        self.expect(
            f"plugin list {plugin_namespace}", substrs=[plugin_namespace, "[+]"]
        )

        # Plugins can be disabled.
        self.expect(
            f"plugin disable {plugin_namespace}", substrs=[plugin_namespace, "[-]"]
        )

        # Plugins can be enabled.
        self.expect(
            f"plugin enable {plugin_namespace}", substrs=[plugin_namespace, "[+]"]
        )
