"""
Minimal scripted frame provider used only to exercise `target
frame-provider register`. get_frame_at_index is never invoked by this
test: registration alone is enough to trigger the bug under test.
"""

from lldb.plugins.scripted_frame_provider import ScriptedFrameProvider


class MinimalProvider(ScriptedFrameProvider):
    @staticmethod
    def get_description():
        return "minimal provider"

    def get_frame_at_index(self, index):
        return None
