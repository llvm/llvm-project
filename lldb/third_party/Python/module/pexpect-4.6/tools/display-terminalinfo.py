#!/usr/bin/env python
""" Display known information about our terminal. """
from __future__ import print_function
import termios
import locale
import sys
import os

BITMAP_IFLAG = {
    'IGNBRK': 'ignore BREAK condition',
    'BRKINT': 'map BREAK to SIGINTR',
    'IGNPAR': 'ignore (discard) parity errors',
    'PARMRK': 'mark parity and framing errors',
    'INPCK': 'enable checking of parity errors',
    'ISTRIP': 'strip 8th bit off chars',
    'INLCR': 'map NL into CR',
    'IGNCR': 'ignore CR',
    'ICRNL': 'map CR to NL (ala CRMOD)',
    'IXON': 'enable output flow control',
    'IXOFF': 'enable input flow control',
    'IXANY': 'any char will restart after stop',
    'IMAXBEL': 'ring bell on input queue full',
    'IUCLC': 'translate upper case to lower case',
}

BITMAP_OFLAG = {
    'OPOST': 'enable following output processing',
    'ONLCR': 'map NL to CR-NL (ala CRMOD)',
    'OXTABS': 'expand tabs to spaces',
    'ONOEOT': 'discard EOT\'s `^D\' on output)',
    'OCRNL': 'map CR to NL',
    'OLCUC': 'translate lower case to upper case',
    'ONOCR': 'No CR output at column 0',
    'ONLRET': 'NL performs CR function',
}

BITMAP_CFLAG = {
    'CSIZE': 'character size mask',
    'CS5': '5 bits (pseudo)',
    'CS6': '6 bits',
    'CS7': '7 bits',
    'CS8': '8 bits',
    'CSTOPB': 'send 2 stop bits',
    'CREAD': 'enable receiver',
    'PARENB': 'parity enable',
    'PARODD': 'odd parity, else even',
    'HUPCL': 'hang up on last close',
    'CLOCAL': 'ignore modem status lines',
    'CCTS_OFLOW': 'CTS flow control of output',
    'CRTSCTS': 'same as CCTS_OFLOW',
    'CRTS_IFLOW': 'RTS flow control of input',
    'MDMBUF': 'flow control output via Carrier',
}

BITMAP_LFLAG = {
    'ECHOKE': 'visual erase for line kill',
    'ECHOE': 'visually erase chars',
    'ECHO': 'enable echoing',
    'ECHONL': 'echo NL even if ECHO is off',
    'ECHOPRT': 'visual erase mode for hardcopy',
    'ECHOCTL': 'echo control chars as ^(Char)',
    'ISIG': 'enable signals INTR, QUIT, [D]SUSP',
    'ICANON': 'canonicalize input lines',
    'ALTWERASE': 'use alternate WERASE algorithm',
    'IEXTEN': 'enable DISCARD and LNEXT',
    'EXTPROC': 'external processing',
    'TOSTOP': 'stop background jobs from output',
    'FLUSHO': 'output being flushed (state)',
    'NOKERNINFO': 'no kernel output from VSTATUS',
    'PENDIN': 'XXX retype pending input (state)',
    'NOFLSH': 'don\'t flush after interrupt',
}

CTLCHAR_INDEX = {
    'VEOF': 'EOF',
    'VEOL': 'EOL',
    'VEOL2': 'EOL2',
    'VERASE': 'ERASE',
    'VWERASE': 'WERASE',
    'VKILL': 'KILL',
    'VREPRINT': 'REPRINT',
    'VINTR': 'INTR',
    'VQUIT': 'QUIT',
    'VSUSP': 'SUSP',
    'VDSUSP': 'DSUSP',
    'VSTART': 'START',
    'VSTOP': 'STOP',
    'VLNEXT': 'LNEXT',
    'VDISCARD': 'DISCARD',
    'VMIN': '---',
    'VTIME': '---',
    'VSTATUS': 'STATUS',
}


def display_bitmask(kind, bitmap, value):
    """ Display all matching bitmask values for ``value`` given ``bitmap``. """
    col1_width = max(map(len, list(bitmap.keys()) + [kind]))
    col2_width = 7
    FMT = '{name:>{col1_width}} {value:>{col2_width}}   {description}'
    print(FMT.format(name=kind,
                     value='Value',
                     description='Description',
                     col1_width=col1_width,
                     col2_width=col2_width))
    print('{0} {1}   {2}'.format('-' * col1_width,
                                 '-' * col2_width,
                                 '-' * max(map(len, bitmap.values()))))
    for flag_name, description in bitmap.items():
        try:
            bitmask = getattr(termios, flag_name)
            bit_val = 'on' if bool(value & bitmask) else 'off'
        except AttributeError:
            bit_val = 'undef'
        print(FMT.format(name=flag_name,
                         value=bit_val,
                         description=description,
                         col1_width=col1_width,
                         col2_width=col2_width))
    print()


def display_ctl_chars(index, cc):
    """ Display all control character indicies, names, and values. """
    title = 'Special Character'
    col1_width = len(title)
    col2_width = max(map(len, index.values()))
    FMT = '{idx:<{col1_width}}   {name:<{col2_width}} {value}'
    print('Special line Characters'.center(40).rstrip())
    print(FMT.format(idx='Index',
                     name='Name',
                     value='Value',
                     col1_width=col1_width,
                     col2_width=col2_width))
    print('{0}   {1} {2}'.format('-' * col1_width,
                                 '-' * col2_width,
                                 '-' * 10))
    for index_name, name in index.items():
        try:
            index = getattr(termios, index_name)
            value = cc[index]
            if value == b'\xff':
                value = '_POSIX_VDISABLE'
            else:
                value = repr(value)
        except AttributeError:
            value = 'undef'
        print(FMT.format(idx=index_name,
                         name=name,
                         value=value,
                         col1_width=col1_width,
                         col2_width=col2_width))
    print()


def display_conf(kind, names, getter):
    col1_width = max(map(len, names))
    FMT = '{name:>{col1_width}}   {value}'
    print(FMT.format(name=kind,
                     value='value',
                     col1_width=col1_width))
    print('{0} {1}'.format('-' * col1_width, '-' * 27))
    for name in names:
        try:
            value = getter(name)
        except OSError as err:
            value = err
        print(FMT.format(name=name, value=value, col1_width=col1_width))
    print()


def main():
    fd = sys.stdin.fileno()
    locale.setlocale(locale.LC_ALL, '')
    encoding = locale.getpreferredencoding()

    print('os.isatty({0}) => {1}'.format(fd, os.isatty(fd)))
    print('locale.getpreferredencoding() => {0}'.format(encoding))

    display_conf(kind='pathconf',
                 names=os.pathconf_names,
                 getter=lambda name: os.fpathconf(fd, name))

    try:
        (iflag, oflag, cflag, lflag, ispeed, ospeed, cc
         ) = termios.tcgetattr(fd)
    except termios.error as err:
        print('stdin is not a typewriter: {0}'.format(err))
    else:
        display_bitmask(kind='Input Mode',
                        bitmap=BITMAP_IFLAG,
                        value=iflag)
        display_bitmask(kind='Output Mode',
                        bitmap=BITMAP_OFLAG,
                        value=oflag)
        display_bitmask(kind='Control Mode',
                        bitmap=BITMAP_CFLAG,
                        value=cflag)
        display_bitmask(kind='Local Mode',
                        bitmap=BITMAP_LFLAG,
                        value=lflag)
        display_ctl_chars(index=CTLCHAR_INDEX,
                          cc=cc)
        try:
            print('os.ttyname({0}) => {1}'.format(fd, os.ttyname(fd)))
            print('os.ctermid() => {0}'.format(os.ctermid()))
        except OSError as e:
            # Travis fails on ttyname with errno 0 'Error'.
            print("Error inspecting TTY: {0}".format(e))

if __name__ == '__main__':
    main()
