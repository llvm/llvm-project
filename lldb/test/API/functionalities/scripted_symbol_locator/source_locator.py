import os
from typing import Optional

import lldb


class SourceLocator:
    """Test locator that records calls and returns a configured resolved path."""

    def __init__(
        self, exe_ctx: lldb.SBExecutionContext, args: lldb.SBStructuredData
    ) -> None:
        self.calls: list = []
        self.resolved_dir: Optional[str] = None
        if args.IsValid():
            resolved_dir_val = args.GetValueForKey("resolved_dir")
            if resolved_dir_val and resolved_dir_val.IsValid():
                val = resolved_dir_val.GetStringValue(4096)
                if val:
                    self.resolved_dir = val

    def locate_source_file(
        self, module: lldb.SBModule, original_source_file: str
    ) -> Optional[lldb.SBFileSpec]:
        uuid = module.GetUUIDString()
        self.calls.append((uuid, original_source_file))
        if self.resolved_dir:
            basename = os.path.basename(original_source_file)
            return lldb.SBFileSpec(os.path.join(self.resolved_dir, basename), True)
        return None


class NoneLocator:
    """Locator that always returns None."""

    def __init__(
        self, exe_ctx: lldb.SBExecutionContext, args: lldb.SBStructuredData
    ) -> None:
        pass

    def locate_source_file(
        self, module: lldb.SBModule, original_source_file: str
    ) -> Optional[lldb.SBFileSpec]:
        return None
