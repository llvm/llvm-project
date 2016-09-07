
# LLDB UI state in the Vim user interface.

import os
import re
import sys
import lldb
import vim
from vim_panes import *
from vim_signs import *


def is_same_file(a, b):
    """ returns true if paths a and b are the same file """
    a = os.path.realpath(a)
    b = os.path.realpath(b)
    return a in b or b in a


class UI:

    def __init__(self):
        """ Declare UI state variables """

        # Default panes to display
        self.defaultPanes = [
            'breakpoints',
            'backtrace',
            'locals',
            'threads',
            'registers',
            'disassembly']

        # map of tuples (filename, line) --> SBBreakpoint
        self.markedBreakpoints = {}

        # Currently shown signs
        self.breakpointSigns = {}
        self.pcSigns = []

        # Container for panes
        self.paneCol = PaneLayout()

        # All possible LLDB panes
        self.backtracePane = BacktracePane(self.paneCol)
        self.threadPane = ThreadPane(self.paneCol)
        self.disassemblyPane = DisassemblyPane(self.paneCol)
        self.localsPane = LocalsPane(self.paneCol)
        self.registersPane = RegistersPane(self.paneCol)
        self.breakPane = BreakpointsPane(self.paneCol)

    def activate(self):
        """ Activate UI: display default set of panes """
        self.paneCol.prepare(self.defaultPanes)

    def get_user_buffers(self, filter_name=None):
        """ Returns a list of buffers that are not a part of the LLDB UI. That is, they
            are not contained in the PaneLayout object self.paneCol.
        """
        ret = []
        for w in vim.windows:
            b = w.buffer
            if not self.paneCol.contains(b.name):
                if filter_name is None or filter_name in b.name:
                    ret.append(b)
        return ret

    def update_pc(self, process, buffers, goto_file):
        """ Place the PC sign on the PC location of each thread's selected frame """

        def GetPCSourceLocation(thread):
            """ Returns a tuple (thread_index, file, line, column) that represents where
                the PC sign should be placed for a thread.
            """

            frame = thread.GetSelectedFrame()
            frame_num = frame.GetFrameID()
            le = frame.GetLineEntry()
            while not le.IsValid() and frame_num < thread.GetNumFrames():
                frame_num += 1
                le = thread.GetFrameAtIndex(frame_num).GetLineEntry()

            if le.IsValid():
                path = os.path.join(
                    le.GetFileSpec().GetDirectory(),
                    le.GetFileSpec().GetFilename())
                return (
                    thread.GetIndexID(),
                    path,
                    le.GetLine(),
                    le.GetColumn())
            return None

        # Clear all existing PC signs
        del_list = []
        for sign in self.pcSigns:
            sign.hide()
            del_list.append(sign)
        for sign in del_list:
            self.pcSigns.remove(sign)
            del sign

        # Select a user (non-lldb) window
        if not self.paneCol.selectWindow(False):
            # No user window found; avoid clobbering by splitting
            vim.command(":vsp")

        # Show a PC marker for each thread
        for thread in process:
            loc = GetPCSourceLocation(thread)
            if not loc:
                # no valid source locations for PCs. hide all existing PC
                # markers
                continue

            buf = None
            (tid, fname, line, col) = loc
            buffers = self.get_user_buffers(fname)
            is_selected = thread.GetIndexID() == process.GetSelectedThread().GetIndexID()
            if len(buffers) == 1:
                buf = buffers[0]
                if buf != vim.current.buffer:
                    # Vim has an open buffer to the required file: select it
                    vim.command('execute ":%db"' % buf.number)
            elif is_selected and vim.current.buffer.name not in fname and os.path.exists(fname) and goto_file:
                # FIXME: If current buffer is modified, vim will complain when we try to switch away.
                # Find a way to detect if the current buffer is modified,
                # and...warn instead?
                vim.command('execute ":e %s"' % fname)
                buf = vim.current.buffer
            elif len(buffers) > 1 and goto_file:
                # FIXME: multiple open buffers match PC location
                continue
            else:
                continue

            self.pcSigns.append(PCSign(buf, line, is_selected))

            if is_selected and goto_file:
                # if the selected file has a PC marker, move the cursor there
                # too
                curname = vim.current.buffer.name
                if curname is not None and is_same_file(curname, fname):
                    move_cursor(line, 0)
                elif move_cursor:
                    print "FIXME: not sure where to move cursor because %s != %s " % (vim.current.buffer.name, fname)

    def update_breakpoints(self, target, buffers):
        """ Decorates buffer with signs corresponding to breakpoints in target. """

        def GetBreakpointLocations(bp):
            """ Returns a list of tuples (resolved, filename, line) where a breakpoint was resolved. """
            if not bp.IsValid():
                sys.stderr.write("breakpoint is invalid, no locations")
                return []

            ret = []
            numLocs = bp.GetNumLocations()
            for i in range(numLocs):
                loc = bp.GetLocationAtIndex(i)
                desc = get_description(loc, lldb.eDescriptionLevelFull)
                match = re.search('at\ ([^:]+):([\d]+)', desc)
                try:
                    lineNum = int(match.group(2).strip())
                    ret.append((loc.IsResolved(), match.group(1), lineNum))
                except ValueError as e:
                    sys.stderr.write(
                        "unable to parse breakpoint location line number: '%s'" %
                        match.group(2))
                    sys.stderr.write(str(e))

            return ret

        if target is None or not target.IsValid():
            return

        needed_bps = {}
        for bp_index in range(target.GetNumBreakpoints()):
            bp = target.GetBreakpointAtIndex(bp_index)
            locations = GetBreakpointLocations(bp)
            for (is_resolved, file, line) in GetBreakpointLocations(bp):
                for buf in buffers:
                    if file in buf.name:
                        needed_bps[(buf, line, is_resolved)] = bp

        # Hide any signs that correspond with disabled breakpoints
        del_list = []
        for (b, l, r) in self.breakpointSigns:
            if (b, l, r) not in needed_bps:
                self.breakpointSigns[(b, l, r)].hide()
                del_list.append((b, l, r))
        for d in del_list:
            del self.breakpointSigns[d]

        # Show any signs for new breakpoints
        for (b, l, r) in needed_bps:
            bp = needed_bps[(b, l, r)]
            if self.haveBreakpoint(b.name, l):
                self.markedBreakpoints[(b.name, l)].append(bp)
            else:
                self.markedBreakpoints[(b.name, l)] = [bp]

            if (b, l, r) not in self.breakpointSigns:
                s = BreakpointSign(b, l, r)
                self.breakpointSigns[(b, l, r)] = s

    def update(self, target, status, controller, goto_file=False):
        """ Updates debugger info panels and breakpoint/pc marks and prints
            status to the vim status line. If goto_file is True, the user's
            cursor is moved to the source PC location in the selected frame.
        """

        self.paneCol.update(target, controller)
        self.update_breakpoints(target, self.get_user_buffers())

        if target is not None and target.IsValid():
            process = target.GetProcess()
            if process is not None and process.IsValid():
                self.update_pc(process, self.get_user_buffers, goto_file)

        if status is not None and len(status) > 0:
            print status

    def haveBreakpoint(self, file, line):
        """ Returns True if we have a breakpoint at file:line, False otherwise  """
        return (file, line) in self.markedBreakpoints

    def getBreakpoints(self, fname, line):
        """ Returns the LLDB SBBreakpoint object at fname:line """
        if self.haveBreakpoint(fname, line):
            return self.markedBreakpoints[(fname, line)]
        else:
            return None

    def deleteBreakpoints(self, name, line):
        del self.markedBreakpoints[(name, line)]

    def showWindow(self, name):
        """ Shows (un-hides) window pane specified by name """
        if not self.paneCol.havePane(name):
            sys.stderr.write("unknown window: %s" % name)
            return False
        self.paneCol.prepare([name])
        return True

    def hideWindow(self, name):
        """ Hides window pane specified by name """
        if not self.paneCol.havePane(name):
            sys.stderr.write("unknown window: %s" % name)
            return False
        self.paneCol.hide([name])
        return True

global ui
ui = UI()
