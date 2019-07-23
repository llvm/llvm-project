#
# This file contains implementations of the LLDB display panes in VIM
#
# The most generic way to define a new window is to inherit from VimPane
# and to implement:
# - get_content() - returns a string with the pane contents
#
# Optionally, to highlight text, implement:
# - get_highlights() - returns a map
#
# And call:
# - define_highlight(unique_name, colour)
# at some point in the constructor.
#
#
# If the pane shows some key-value data that is in the context of a
# single frame, inherit from FrameKeyValuePane and implement:
# - get_frame_content(self, SBFrame frame)
#
#
# If the pane presents some information that can be retrieved with
# a simple LLDB command while the subprocess is stopped, inherit
# from StoppedCommandPane and call:
# - self.setCommand(command, command_args)
# at some point in the constructor.
#
# Optionally, you can implement:
# - get_selected_line()
# to highlight a selected line and place the cursor there.
#
#
# FIXME: implement WatchlistPane to displayed watched expressions
# FIXME: define interface for interactive panes, like catching enter
#        presses to change selected frame/thread...
#

import lldb
import vim

import sys

# ==============================================================
# Get the description of an lldb object or None if not available
# ==============================================================

# Shamelessly copy/pasted from lldbutil.py in the test suite


def get_description(obj, option=None):
    """Calls lldb_obj.GetDescription() and returns a string, or None.

    For SBTarget, SBBreakpointLocation, and SBWatchpoint lldb objects, an extra
    option can be passed in to describe the detailed level of description
    desired:
        o lldb.eDescriptionLevelBrief
        o lldb.eDescriptionLevelFull
        o lldb.eDescriptionLevelVerbose
    """
    method = getattr(obj, 'GetDescription')
    if not method:
        return None
    tuple = (lldb.SBTarget, lldb.SBBreakpointLocation, lldb.SBWatchpoint)
    if isinstance(obj, tuple):
        if option is None:
            option = lldb.eDescriptionLevelBrief

    stream = lldb.SBStream()
    if option is None:
        success = method(stream)
    else:
        success = method(stream, option)
    if not success:
        return None
    return stream.GetData()


def get_selected_thread(target):
    """ Returns a tuple with (thread, error) where thread == None if error occurs """
    process = target.GetProcess()
    if process is None or not process.IsValid():
        return (None, VimPane.MSG_NO_PROCESS)

    thread = process.GetSelectedThread()
    if thread is None or not thread.IsValid():
        return (None, VimPane.MSG_NO_THREADS)
    return (thread, "")


def get_selected_frame(target):
    """ Returns a tuple with (frame, error) where frame == None if error occurs """
    (thread, error) = get_selected_thread(target)
    if thread is None:
        return (None, error)

    frame = thread.GetSelectedFrame()
    if frame is None or not frame.IsValid():
        return (None, VimPane.MSG_NO_FRAME)
    return (frame, "")


def _cmd(cmd):
    vim.command("call confirm('%s')" % cmd)
    vim.command(cmd)


def move_cursor(line, col=0):
    """ moves cursor to specified line and col """
    cw = vim.current.window
    if cw.cursor[0] != line:
        vim.command("execute \"normal %dgg\"" % line)


def winnr():
    """ Returns currently selected window number """
    return int(vim.eval("winnr()"))


def bufwinnr(name):
    """ Returns window number corresponding with buffer name """
    return int(vim.eval("bufwinnr('%s')" % name))


def goto_window(nr):
    """ go to window number nr"""
    if nr != winnr():
        vim.command(str(nr) + ' wincmd w')


def goto_next_window():
    """ go to next window. """
    vim.command('wincmd w')
    return (winnr(), vim.current.buffer.name)


def goto_previous_window():
    """ go to previously selected window """
    vim.command("execute \"normal \\<c-w>p\"")


def have_gui():
    """ Returns True if vim is in a gui (Gvim/MacVim), False otherwise. """
    return int(vim.eval("has('gui_running')")) == 1


class PaneLayout(object):
    """ A container for a (vertical) group layout of VimPanes """

    def __init__(self):
        self.panes = {}

    def havePane(self, name):
        """ Returns true if name is a registered pane, False otherwise """
        return name in self.panes

    def prepare(self, panes=[]):
        """ Draw panes on screen. If empty list is provided, show all. """

        # If we can't select a window contained in the layout, we are doing a
        # first draw
        first_draw = not self.selectWindow(True)
        did_first_draw = False

        # Prepare each registered pane
        for name in self.panes:
            if name in panes or len(panes) == 0:
                if first_draw:
                    # First window in layout will be created with :vsp, and
                    # closed later
                    vim.command(":vsp")
                    first_draw = False
                    did_first_draw = True
                self.panes[name].prepare()

        if did_first_draw:
            # Close the split window
            vim.command(":q")

        self.selectWindow(False)

    def contains(self, bufferName=None):
        """ Returns True if window with name bufferName is contained in the layout, False otherwise.
            If bufferName is None, the currently selected window is checked.
        """
        if not bufferName:
            bufferName = vim.current.buffer.name

        for p in self.panes:
            if bufferName is not None and bufferName.endswith(p):
                return True
        return False

    def selectWindow(self, select_contained=True):
        """ Selects a window contained in the layout (if select_contained = True) and returns True.
            If select_contained = False, a window that is not contained is selected. Returns False
            if no group windows can be selected.
        """
        if select_contained == self.contains():
            # Simple case: we are already selected
            return True

        # Otherwise, switch to next window until we find a contained window, or
        # reach the first window again.
        first = winnr()
        (curnum, curname) = goto_next_window()

        while not select_contained == self.contains(
                curname) and curnum != first:
            (curnum, curname) = goto_next_window()

        return self.contains(curname) == select_contained

    def hide(self, panes=[]):
        """ Hide panes specified. If empty list provided, hide all. """
        for name in self.panes:
            if name in panes or len(panes) == 0:
                self.panes[name].destroy()

    def registerForUpdates(self, p):
        self.panes[p.name] = p

    def update(self, target, controller):
        for name in self.panes:
            self.panes[name].update(target, controller)


class VimPane(object):
    """ A generic base class for a pane that displays stuff """
    CHANGED_VALUE_HIGHLIGHT_NAME_GUI = 'ColorColumn'
    CHANGED_VALUE_HIGHLIGHT_NAME_TERM = 'lldb_changed'
    CHANGED_VALUE_HIGHLIGHT_COLOUR_TERM = 'darkred'

    SELECTED_HIGHLIGHT_NAME_GUI = 'Cursor'
    SELECTED_HIGHLIGHT_NAME_TERM = 'lldb_selected'
    SELECTED_HIGHLIGHT_COLOUR_TERM = 'darkblue'

    MSG_NO_TARGET = "Target does not exist."
    MSG_NO_PROCESS = "Process does not exist."
    MSG_NO_THREADS = "No valid threads."
    MSG_NO_FRAME = "No valid frame."

    # list of defined highlights, so we avoid re-defining them
    highlightTypes = []

    def __init__(self, owner, name, open_below=False, height=3):
        self.owner = owner
        self.name = name
        self.buffer = None
        self.maxHeight = 20
        self.openBelow = open_below
        self.height = height
        self.owner.registerForUpdates(self)

    def isPrepared(self):
        """ check window is OK """
        if self.buffer is None or len(
                dir(self.buffer)) == 0 or bufwinnr(self.name) == -1:
            return False
        return True

    def prepare(self, method='new'):
        """ check window is OK, if not then create """
        if not self.isPrepared():
            self.create(method)

    def on_create(self):
        pass

    def destroy(self):
        """ destroy window """
        if self.buffer is None or len(dir(self.buffer)) == 0:
            return
        vim.command('bdelete ' + self.name)

    def create(self, method):
        """ create window """

        if method != 'edit':
            belowcmd = "below" if self.openBelow else ""
            vim.command('silent %s %s %s' % (belowcmd, method, self.name))
        else:
            vim.command('silent %s %s' % (method, self.name))

        self.window = vim.current.window

        # Set LLDB pane options
        vim.command("setlocal buftype=nofile")  # Don't try to open a file
        vim.command("setlocal noswapfile")     # Don't use a swap file
        vim.command("set nonumber")            # Don't display line numbers
        # vim.command("set nowrap")              # Don't wrap text

        # Save some parameters and reference to buffer
        self.buffer = vim.current.buffer
        self.width = int(vim.eval("winwidth(0)"))
        self.height = int(vim.eval("winheight(0)"))

        self.on_create()
        goto_previous_window()

    def update(self, target, controller):
        """ updates buffer contents """
        self.target = target
        if not self.isPrepared():
            # Window is hidden, or otherwise not ready for an update
            return

        original_cursor = self.window.cursor

        # Select pane
        goto_window(bufwinnr(self.name))

        # Clean and update content, and apply any highlights.
        self.clean()

        if self.write(self.get_content(target, controller)):
            self.apply_highlights()

            cursor = self.get_selected_line()
            if cursor is None:
                # Place the cursor at its original position in the window
                cursor_line = min(original_cursor[0], len(self.buffer))
                cursor_col = min(
                    original_cursor[1], len(
                        self.buffer[
                            cursor_line - 1]))
            else:
                # Place the cursor at the location requested by a VimPane
                # implementation
                cursor_line = min(cursor, len(self.buffer))
                cursor_col = self.window.cursor[1]

            self.window.cursor = (cursor_line, cursor_col)

        goto_previous_window()

    def get_selected_line(self):
        """ Returns the line number to move the cursor to, or None to leave
            it where the user last left it.
            Subclasses implement this to define custom behaviour.
        """
        return None

    def apply_highlights(self):
        """ Highlights each set of lines in  each highlight group """
        highlights = self.get_highlights()
        for highlightType in highlights:
            lines = highlights[highlightType]
            if len(lines) == 0:
                continue

            cmd = 'match %s /' % highlightType
            lines = ['\%' + '%d' % line + 'l' for line in lines]
            cmd += '\\|'.join(lines)
            cmd += '/'
            vim.command(cmd)

    def define_highlight(self, name, colour):
        """ Defines highlihght """
        if name in VimPane.highlightTypes:
            # highlight already defined
            return

        vim.command(
            "highlight %s ctermbg=%s guibg=%s" %
            (name, colour, colour))
        VimPane.highlightTypes.append(name)

    def write(self, msg):
        """ replace buffer with msg"""
        self.prepare()

        msg = str(msg.encode("utf-8", "replace")).split('\n')
        try:
            self.buffer.append(msg)
            vim.command("execute \"normal ggdd\"")
        except vim.error:
            # cannot update window; happens when vim is exiting.
            return False

        move_cursor(1, 0)
        return True

    def clean(self):
        """ clean all datas in buffer """
        self.prepare()
        vim.command(':%d')
        #self.buffer[:] = None

    def get_content(self, target, controller):
        """ subclasses implement this to provide pane content """
        assert(0 and "pane subclass must implement this")
        pass

    def get_highlights(self):
        """ Subclasses implement this to provide pane highlights.
            This function is expected to return a map of:
              { highlight_name ==> [line_number, ...], ... }
        """
        return {}


class FrameKeyValuePane(VimPane):

    def __init__(self, owner, name, open_below):
        """ Initialize parent, define member variables, choose which highlight
            to use based on whether or not we have a gui (MacVim/Gvim).
        """

        VimPane.__init__(self, owner, name, open_below)

        # Map-of-maps key/value history { frame --> { variable_name,
        # variable_value } }
        self.frameValues = {}

        if have_gui():
            self.changedHighlight = VimPane.CHANGED_VALUE_HIGHLIGHT_NAME_GUI
        else:
            self.changedHighlight = VimPane.CHANGED_VALUE_HIGHLIGHT_NAME_TERM
            self.define_highlight(VimPane.CHANGED_VALUE_HIGHLIGHT_NAME_TERM,
                                  VimPane.CHANGED_VALUE_HIGHLIGHT_COLOUR_TERM)

    def format_pair(self, key, value, changed=False):
        """ Formats a key/value pair. Appends a '*' if changed == True """
        marker = '*' if changed else ' '
        return "%s %s = %s\n" % (marker, key, value)

    def get_content(self, target, controller):
        """ Get content for a frame-aware pane. Also builds the list of lines that
            need highlighting (i.e. changed values.)
        """
        if target is None or not target.IsValid():
            return VimPane.MSG_NO_TARGET

        self.changedLines = []

        (frame, err) = get_selected_frame(target)
        if frame is None:
            return err

        output = get_description(frame)
        lineNum = 1

        # Retrieve the last values displayed for this frame
        frameId = get_description(frame.GetBlock())
        if frameId in self.frameValues:
            frameOldValues = self.frameValues[frameId]
        else:
            frameOldValues = {}

        # Read the frame variables
        vals = self.get_frame_content(frame)
        for (key, value) in vals:
            lineNum += 1
            if len(frameOldValues) == 0 or (
                    key in frameOldValues and frameOldValues[key] == value):
                output += self.format_pair(key, value)
            else:
                output += self.format_pair(key, value, True)
                self.changedLines.append(lineNum)

        # Save values as oldValues
        newValues = {}
        for (key, value) in vals:
            newValues[key] = value
        self.frameValues[frameId] = newValues

        return output

    def get_highlights(self):
        ret = {}
        ret[self.changedHighlight] = self.changedLines
        return ret


class LocalsPane(FrameKeyValuePane):
    """ Pane that displays local variables """

    def __init__(self, owner, name='locals'):
        FrameKeyValuePane.__init__(self, owner, name, open_below=True)

        # FIXME: allow users to customize display of args/locals/statics/scope
        self.arguments = True
        self.show_locals = True
        self.show_statics = True
        self.show_in_scope_only = True

    def format_variable(self, var):
        """ Returns a Tuple of strings "(Type) Name", "Value" for SBValue var """
        val = var.GetValue()
        if val is None:
            # If the value is too big, SBValue.GetValue() returns None; replace
            # with ...
            val = "..."

        return ("(%s) %s" % (var.GetTypeName(), var.GetName()), "%s" % val)

    def get_frame_content(self, frame):
        """ Returns list of key-value pairs of local variables in frame """
        vals = frame.GetVariables(self.arguments,
                                  self.show_locals,
                                  self.show_statics,
                                  self.show_in_scope_only)
        return [self.format_variable(x) for x in vals]


class RegistersPane(FrameKeyValuePane):
    """ Pane that displays the contents of registers """

    def __init__(self, owner, name='registers'):
        FrameKeyValuePane.__init__(self, owner, name, open_below=True)

    def format_register(self, reg):
        """ Returns a tuple of strings ("name", "value") for SBRegister reg. """
        name = reg.GetName()
        val = reg.GetValue()
        if val is None:
            val = "..."
        return (name, val.strip())

    def get_frame_content(self, frame):
        """ Returns a list of key-value pairs ("name", "value") of registers in frame """

        result = []
        for register_sets in frame.GetRegisters():
            # hack the register group name into the list of registers...
            result.append((" = = %s =" % register_sets.GetName(), ""))

            for reg in register_sets:
                result.append(self.format_register(reg))
        return result


class CommandPane(VimPane):
    """ Pane that displays the output of an LLDB command """

    def __init__(self, owner, name, open_below, process_required=True):
        VimPane.__init__(self, owner, name, open_below)
        self.process_required = process_required

    def setCommand(self, command, args=""):
        self.command = command
        self.args = args

    def get_content(self, target, controller):
        output = ""
        if not target:
            output = VimPane.MSG_NO_TARGET
        elif self.process_required and not target.GetProcess():
            output = VimPane.MSG_NO_PROCESS
        else:
            (success, output) = controller.getCommandOutput(
                self.command, self.args)
        return output


class StoppedCommandPane(CommandPane):
    """ Pane that displays the output of an LLDB command when the process is
        stopped; otherwise displays process status. This class also implements
        highlighting for a single line (to show a single-line selected entity.)
    """

    def __init__(self, owner, name, open_below):
        """ Initialize parent and define highlight to use for selected line. """
        CommandPane.__init__(self, owner, name, open_below)
        if have_gui():
            self.selectedHighlight = VimPane.SELECTED_HIGHLIGHT_NAME_GUI
        else:
            self.selectedHighlight = VimPane.SELECTED_HIGHLIGHT_NAME_TERM
            self.define_highlight(VimPane.SELECTED_HIGHLIGHT_NAME_TERM,
                                  VimPane.SELECTED_HIGHLIGHT_COLOUR_TERM)

    def get_content(self, target, controller):
        """ Returns the output of a command that relies on the process being stopped.
            If the process is not in 'stopped' state, the process status is returned.
        """
        output = ""
        if not target or not target.IsValid():
            output = VimPane.MSG_NO_TARGET
        elif not target.GetProcess() or not target.GetProcess().IsValid():
            output = VimPane.MSG_NO_PROCESS
        elif target.GetProcess().GetState() == lldb.eStateStopped:
            (success, output) = controller.getCommandOutput(
                self.command, self.args)
        else:
            (success, output) = controller.getCommandOutput("process", "status")
        return output

    def get_highlights(self):
        """ Highlight the line under the cursor. Users moving the cursor has
            no effect on the selected line.
        """
        ret = {}
        line = self.get_selected_line()
        if line is not None:
            ret[self.selectedHighlight] = [line]
            return ret
        return ret

    def get_selected_line(self):
        """ Subclasses implement this to control where the cursor (and selected highlight)
            is placed.
        """
        return None


class DisassemblyPane(CommandPane):
    """ Pane that displays disassembly around PC """

    def __init__(self, owner, name='disassembly'):
        CommandPane.__init__(self, owner, name, open_below=True)

        # FIXME: let users customize the number of instructions to disassemble
        self.setCommand("disassemble", "-c %d -p" % self.maxHeight)


class ThreadPane(StoppedCommandPane):
    """ Pane that displays threads list """

    def __init__(self, owner, name='threads'):
        StoppedCommandPane.__init__(self, owner, name, open_below=False)
        self.setCommand("thread", "list")

# FIXME: the function below assumes threads are listed in sequential order,
#        which turns out to not be the case. Highlighting of selected thread
#        will be disabled until this can be fixed. LLDB prints a '*' anyways
#        beside the selected thread, so this is not too big of a problem.
#  def get_selected_line(self):
#    """ Place the cursor on the line with the selected entity.
#        Subclasses should override this to customize selection.
#        Formula: selected_line = selected_thread_id + 1
#    """
#    (thread, err) = get_selected_thread(self.target)
#    if thread is None:
#      return None
#    else:
#      return thread.GetIndexID() + 1


class BacktracePane(StoppedCommandPane):
    """ Pane that displays backtrace """

    def __init__(self, owner, name='backtrace'):
        StoppedCommandPane.__init__(self, owner, name, open_below=False)
        self.setCommand("bt", "")

    def get_selected_line(self):
        """ Returns the line number in the buffer with the selected frame.
            Formula: selected_line = selected_frame_id + 2
            FIXME: the above formula hack does not work when the function return
                   value is printed in the bt window; the wrong line is highlighted.
        """

        (frame, err) = get_selected_frame(self.target)
        if frame is None:
            return None
        else:
            return frame.GetFrameID() + 2


class BreakpointsPane(CommandPane):

    def __init__(self, owner, name='breakpoints'):
        super(
            BreakpointsPane,
            self).__init__(
            owner,
            name,
            open_below=False,
            process_required=False)
        self.setCommand("breakpoint", "list")
