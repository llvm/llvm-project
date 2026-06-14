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

    def test_completions(self):
        # Make sure completions work for the plugin list, enable, and disable commands.
        # We just check a few of the expected plugins to make sure the completion works.
        self.completions_contain(
            "plugin list ", ["abi", "architecture", "disassembler"]
        )
        self.completions_contain(
            "plugin enable ", ["abi", "architecture", "disassembler"]
        )
        self.completions_contain(
            "plugin disable ", ["abi", "architecture", "disassembler"]
        )

        # A completion for a partial namespace should be the full namespace.
        # This allows the user to run the command on the full namespace.
        self.completions_match("plugin list ab", ["abi"])
        self.completions_contain(
            "plugin list object", ["object-container", "object-file"]
        )

        # A completion for a full namespace should contain the plugins in that namespace.
        self.completions_contain("plugin list object-file", ["object-file.JSON"])
        self.completions_contain("plugin list object-file.", ["object-file.JSON"])
        self.completions_contain("plugin list object-file.J", ["object-file.JSON"])
        self.completions_contain("plugin list object-file.JS", ["object-file.JSON"])

        # Check for a completion that is a both a complete namespace and a prefix of
        # another namespace. It should return the completions for the plugins in the completed
        # namespace as well as the completion for the partial namespace.
        self.completions_contain(
            "plugin list language", ["language.cplusplus", "language-runtime"]
        )

        # When the namespace is a prefix of another namespace and the user types a dot, the
        # completion should not include the match for the partial namespace.
        self.completions_contain(
            "plugin list language.", ["language.cplusplus"], match=True
        )
        self.completions_contain(
            "plugin list language.", ["language-runtime"], match=False
        )

        # Check for an empty completion list when the names is invalid.
        # See docs for `complete_from_to` for how this checks for an empty list.
        self.complete_from_to("plugin list abi.foo", ["plugin list abi.foo"])
