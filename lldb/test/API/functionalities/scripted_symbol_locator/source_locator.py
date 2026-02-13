import os

import lldb


class SourceLocator:
    """Test locator that records calls and returns a configured resolved path."""

    calls = []
    resolved_dir = None

    def __init__(self, exe_ctx, args):
        SourceLocator.calls = []

    def locate_source_file(self, module, original_source_file):
        uuid = module.GetUUIDString()
        SourceLocator.calls.append((uuid, original_source_file))
        if SourceLocator.resolved_dir:
            basename = os.path.basename(original_source_file)
            return os.path.join(SourceLocator.resolved_dir, basename)
        return None

    def locate_executable_object_file(self, module_spec):
        return None

    def locate_executable_symbol_file(self, module_spec, default_search_paths):
        return None

    def download_object_and_symbol_file(self, module_spec, force_lookup,
                                        copy_executable):
        return False


class NoneLocator:
    """Locator that always returns None."""

    def __init__(self, exe_ctx, args):
        pass

    def locate_source_file(self, module, original_source_file):
        return None

    def locate_executable_object_file(self, module_spec):
        return None

    def locate_executable_symbol_file(self, module_spec, default_search_paths):
        return None

    def download_object_and_symbol_file(self, module_spec, force_lookup,
                                        copy_executable):
        return False
