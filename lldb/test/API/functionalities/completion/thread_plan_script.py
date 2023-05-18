#############################################################################
# This script is just to provide a thread plan which won't be popped instantly
# for the completion test. The thread plan class below won't really do anything
# itself.

import lldb


class PushPlanStack:
    def __init__(self, thread_plan, dict):
        pass

    def explains_stop(self, event):
        return False

    def should_stop(self, event):
        return True

    def should_step(self):
        return True
