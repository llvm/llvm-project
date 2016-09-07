##===-- eventwin.py ------------------------------------------*- Python -*-===##
##
# The LLVM Compiler Infrastructure
##
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
##
##===----------------------------------------------------------------------===##

import cui
import lldb
import lldbutil


class EventWin(cui.TitledWin):

    def __init__(self, x, y, w, h):
        super(EventWin, self).__init__(x, y, w, h, 'LLDB Event Log')
        self.win.scrollok(1)
        super(EventWin, self).draw()

    def handleEvent(self, event):
        if isinstance(event, lldb.SBEvent):
            self.win.scroll()
            h = self.win.getmaxyx()[0]
            self.win.addstr(h - 1, 0, lldbutil.get_description(event))
        return
