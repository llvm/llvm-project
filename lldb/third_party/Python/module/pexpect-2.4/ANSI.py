"""This implements an ANSI terminal emulator as a subclass of screen.

$Id: ANSI.py 491 2007-12-16 20:04:57Z noah $
"""
# references:
#     http://www.retards.org/terminals/vt102.html
#     http://vt100.net/docs/vt102-ug/contents.html
#     http://vt100.net/docs/vt220-rm/
#     http://www.termsys.demon.co.uk/vtansi.htm

import screen
import FSM
import copy
import string


def Emit(fsm):

    screen = fsm.memory[0]
    screen.write_ch(fsm.input_symbol)


def StartNumber(fsm):

    fsm.memory.append(fsm.input_symbol)


def BuildNumber(fsm):

    ns = fsm.memory.pop()
    ns = ns + fsm.input_symbol
    fsm.memory.append(ns)


def DoBackOne(fsm):

    screen = fsm.memory[0]
    screen.cursor_back()


def DoBack(fsm):

    count = int(fsm.memory.pop())
    screen = fsm.memory[0]
    screen.cursor_back(count)


def DoDownOne(fsm):

    screen = fsm.memory[0]
    screen.cursor_down()


def DoDown(fsm):

    count = int(fsm.memory.pop())
    screen = fsm.memory[0]
    screen.cursor_down(count)


def DoForwardOne(fsm):

    screen = fsm.memory[0]
    screen.cursor_forward()


def DoForward(fsm):

    count = int(fsm.memory.pop())
    screen = fsm.memory[0]
    screen.cursor_forward(count)


def DoUpReverse(fsm):

    screen = fsm.memory[0]
    screen.cursor_up_reverse()


def DoUpOne(fsm):

    screen = fsm.memory[0]
    screen.cursor_up()


def DoUp(fsm):

    count = int(fsm.memory.pop())
    screen = fsm.memory[0]
    screen.cursor_up(count)


def DoHome(fsm):

    c = int(fsm.memory.pop())
    r = int(fsm.memory.pop())
    screen = fsm.memory[0]
    screen.cursor_home(r, c)


def DoHomeOrigin(fsm):

    c = 1
    r = 1
    screen = fsm.memory[0]
    screen.cursor_home(r, c)


def DoEraseDown(fsm):

    screen = fsm.memory[0]
    screen.erase_down()


def DoErase(fsm):

    arg = int(fsm.memory.pop())
    screen = fsm.memory[0]
    if arg == 0:
        screen.erase_down()
    elif arg == 1:
        screen.erase_up()
    elif arg == 2:
        screen.erase_screen()


def DoEraseEndOfLine(fsm):

    screen = fsm.memory[0]
    screen.erase_end_of_line()


def DoEraseLine(fsm):

    screen = fsm.memory[0]
    if arg == 0:
        screen.end_of_line()
    elif arg == 1:
        screen.start_of_line()
    elif arg == 2:
        screen.erase_line()


def DoEnableScroll(fsm):

    screen = fsm.memory[0]
    screen.scroll_screen()


def DoCursorSave(fsm):

    screen = fsm.memory[0]
    screen.cursor_save_attrs()


def DoCursorRestore(fsm):

    screen = fsm.memory[0]
    screen.cursor_restore_attrs()


def DoScrollRegion(fsm):

    screen = fsm.memory[0]
    r2 = int(fsm.memory.pop())
    r1 = int(fsm.memory.pop())
    screen.scroll_screen_rows(r1, r2)


def DoMode(fsm):

    screen = fsm.memory[0]
    mode = fsm.memory.pop()  # Should be 4
    # screen.setReplaceMode ()


def Log(fsm):

    screen = fsm.memory[0]
    fsm.memory = [screen]
    fout = open('log', 'a')
    fout.write(fsm.input_symbol + ',' + fsm.current_state + '\n')
    fout.close()


class term (screen.screen):
    """This is a placeholder.
    In theory I might want to add other terminal types.
    """

    def __init__(self, r=24, c=80):
        screen.screen.__init__(self, r, c)


class ANSI (term):

    """This class encapsulates a generic terminal. It filters a stream and
    maintains the state of a screen object. """

    def __init__(self, r=24, c=80):

        term.__init__(self, r, c)

        #self.screen = screen (24,80)
        self.state = FSM.FSM('INIT', [self])
        self.state.set_default_transition(Log, 'INIT')
        self.state.add_transition_any('INIT', Emit, 'INIT')
        self.state.add_transition('\x1b', 'INIT', None, 'ESC')
        self.state.add_transition_any('ESC', Log, 'INIT')
        self.state.add_transition('(', 'ESC', None, 'G0SCS')
        self.state.add_transition(')', 'ESC', None, 'G1SCS')
        self.state.add_transition_list('AB012', 'G0SCS', None, 'INIT')
        self.state.add_transition_list('AB012', 'G1SCS', None, 'INIT')
        self.state.add_transition('7', 'ESC', DoCursorSave, 'INIT')
        self.state.add_transition('8', 'ESC', DoCursorRestore, 'INIT')
        self.state.add_transition('M', 'ESC', DoUpReverse, 'INIT')
        self.state.add_transition('>', 'ESC', DoUpReverse, 'INIT')
        self.state.add_transition('<', 'ESC', DoUpReverse, 'INIT')
        # Selects application keypad.
        self.state.add_transition('=', 'ESC', None, 'INIT')
        self.state.add_transition('#', 'ESC', None, 'GRAPHICS_POUND')
        self.state.add_transition_any('GRAPHICS_POUND', None, 'INIT')
        self.state.add_transition('[', 'ESC', None, 'ELB')
        # ELB means Escape Left Bracket. That is ^[[
        self.state.add_transition('H', 'ELB', DoHomeOrigin, 'INIT')
        self.state.add_transition('D', 'ELB', DoBackOne, 'INIT')
        self.state.add_transition('B', 'ELB', DoDownOne, 'INIT')
        self.state.add_transition('C', 'ELB', DoForwardOne, 'INIT')
        self.state.add_transition('A', 'ELB', DoUpOne, 'INIT')
        self.state.add_transition('J', 'ELB', DoEraseDown, 'INIT')
        self.state.add_transition('K', 'ELB', DoEraseEndOfLine, 'INIT')
        self.state.add_transition('r', 'ELB', DoEnableScroll, 'INIT')
        self.state.add_transition('m', 'ELB', None, 'INIT')
        self.state.add_transition('?', 'ELB', None, 'MODECRAP')
        self.state.add_transition_list(
            string.digits, 'ELB', StartNumber, 'NUMBER_1')
        self.state.add_transition_list(
            string.digits, 'NUMBER_1', BuildNumber, 'NUMBER_1')
        self.state.add_transition('D', 'NUMBER_1', DoBack, 'INIT')
        self.state.add_transition('B', 'NUMBER_1', DoDown, 'INIT')
        self.state.add_transition('C', 'NUMBER_1', DoForward, 'INIT')
        self.state.add_transition('A', 'NUMBER_1', DoUp, 'INIT')
        self.state.add_transition('J', 'NUMBER_1', DoErase, 'INIT')
        self.state.add_transition('K', 'NUMBER_1', DoEraseLine, 'INIT')
        self.state.add_transition('l', 'NUMBER_1', DoMode, 'INIT')
        # It gets worse... the 'm' code can have infinite number of
        # number;number;number before it. I've never seen more than two,
        # but the specs say it's allowed. crap!
        self.state.add_transition('m', 'NUMBER_1', None, 'INIT')
        # LED control. Same problem as 'm' code.
        self.state.add_transition('q', 'NUMBER_1', None, 'INIT')

        # \E[?47h appears to be "switch to alternate screen"
        # \E[?47l restores alternate screen... I think.
        self.state.add_transition_list(
            string.digits, 'MODECRAP', StartNumber, 'MODECRAP_NUM')
        self.state.add_transition_list(
            string.digits,
            'MODECRAP_NUM',
            BuildNumber,
            'MODECRAP_NUM')
        self.state.add_transition('l', 'MODECRAP_NUM', None, 'INIT')
        self.state.add_transition('h', 'MODECRAP_NUM', None, 'INIT')

# RM   Reset Mode                Esc [ Ps l                   none
        self.state.add_transition(';', 'NUMBER_1', None, 'SEMICOLON')
        self.state.add_transition_any('SEMICOLON', Log, 'INIT')
        self.state.add_transition_list(
            string.digits, 'SEMICOLON', StartNumber, 'NUMBER_2')
        self.state.add_transition_list(
            string.digits, 'NUMBER_2', BuildNumber, 'NUMBER_2')
        self.state.add_transition_any('NUMBER_2', Log, 'INIT')
        self.state.add_transition('H', 'NUMBER_2', DoHome, 'INIT')
        self.state.add_transition('f', 'NUMBER_2', DoHome, 'INIT')
        self.state.add_transition('r', 'NUMBER_2', DoScrollRegion, 'INIT')
        # It gets worse... the 'm' code can have infinite number of
        # number;number;number before it. I've never seen more than two,
        # but the specs say it's allowed. crap!
        self.state.add_transition('m', 'NUMBER_2', None, 'INIT')
        # LED control. Same problem as 'm' code.
        self.state.add_transition('q', 'NUMBER_2', None, 'INIT')

    def process(self, c):

        self.state.process(c)

    def process_list(self, l):

        self.write(l)

    def write(self, s):

        for c in s:
            self.process(c)

    def flush(self):

        pass

    def write_ch(self, ch):
        """This puts a character at the current cursor position. cursor
        position if moved forward with wrap-around, but no scrolling is done if
        the cursor hits the lower-right corner of the screen. """

        #\r and \n both produce a call to crlf().
        ch = ch[0]

        if ch == '\r':
            #    self.crlf()
            return
        if ch == '\n':
            self.crlf()
            return
        if ch == chr(screen.BS):
            self.cursor_back()
            self.put_abs(self.cur_r, self.cur_c, ' ')
            return

        if ch not in string.printable:
            fout = open('log', 'a')
            fout.write('Nonprint: ' + str(ord(ch)) + '\n')
            fout.close()
            return
        self.put_abs(self.cur_r, self.cur_c, ch)
        old_r = self.cur_r
        old_c = self.cur_c
        self.cursor_forward()
        if old_c == self.cur_c:
            self.cursor_down()
            if old_r != self.cur_r:
                self.cursor_home(self.cur_r, 1)
            else:
                self.scroll_up()
                self.cursor_home(self.cur_r, 1)
                self.erase_line()

#    def test (self):
#
#        import sys
#        write_text = 'I\'ve got a ferret sticking up my nose.\n' + \
#        '(He\'s got a ferret sticking up his nose.)\n' + \
#        'How it got there I can\'t tell\n' + \
#        'But now it\'s there it hurts like hell\n' + \
#        'And what is more it radically affects my sense of smell.\n' + \
#        '(His sense of smell.)\n' + \
#        'I can see a bare-bottomed mandril.\n' + \
#        '(Slyly eyeing his other nostril.)\n' + \
#        'If it jumps inside there too I really don\'t know what to do\n' + \
#        'I\'ll be the proud posessor of a kind of nasal zoo.\n' + \
#        '(A nasal zoo.)\n' + \
#        'I\'ve got a ferret sticking up my nose.\n' + \
#        '(And what is worst of all it constantly explodes.)\n' + \
#        '"Ferrets don\'t explode," you say\n' + \
#        'But it happened nine times yesterday\n' + \
#        'And I should know for each time I was standing in the way.\n' + \
#        'I\'ve got a ferret sticking up my nose.\n' + \
#        '(He\'s got a ferret sticking up his nose.)\n' + \
#        'How it got there I can\'t tell\n' + \
#        'But now it\'s there it hurts like hell\n' + \
#        'And what is more it radically affects my sense of smell.\n' + \
#        '(His sense of smell.)'
#        self.fill('.')
#        self.cursor_home()
#        for c in write_text:
#            self.write_ch (c)
#        print str(self)
#
# if __name__ == '__main__':
#    t = ANSI(6,65)
#    t.test()
