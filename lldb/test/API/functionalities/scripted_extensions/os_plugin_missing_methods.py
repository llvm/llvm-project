"""
Standalone OperatingSystem plugin module for
TestScriptedExtensionsDiagnostics.test_operating_system_missing_methods.

`OperatingSystemPython` always resolves the plugin class as
`<module>.OperatingSystemPlugIn`, derived from the
`target.process.python-os-plugin-path` setting's basename, so this scenario
needs its own file rather than a class in malformed_scripted_extensions.py.
"""


class OperatingSystemPlugIn:
    """Missing required abstract method `get_thread_info`."""

    def __init__(self, process):
        self.process = process
