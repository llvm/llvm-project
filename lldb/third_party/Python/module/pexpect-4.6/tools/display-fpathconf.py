#!/usr/bin/env python
"""Displays os.fpathconf values related to terminals. """
from __future__ import print_function
import sys
import os


def display_fpathconf():
    DISP_VALUES = (
        ('PC_MAX_CANON', ('Max no. of bytes in a '
                          'terminal canonical input line.')),
        ('PC_MAX_INPUT', ('Max no. of bytes for which '
                          'space is available in a terminal input queue.')),
        ('PC_PIPE_BUF', ('Max no. of bytes which will '
                         'be written atomically to a pipe.')),
        ('PC_VDISABLE', 'Terminal character disabling value.')
    )
    FMT = '{name:<13} {value:<5} {description}'

    # column header
    print(FMT.format(name='name', value='value', description='description'))
    print(FMT.format(name=('-' * 13), value=('-' * 5), description=('-' * 11)))

    fd = sys.stdin.fileno()
    for name, description in DISP_VALUES:
        key = os.pathconf_names.get(name, None)
        if key is None:
            value = 'UNDEF'
        else:
            try:
                value = os.fpathconf(fd, name)
            except OSError as err:
                value = 'OSErrno {0.errno}'.format(err)
        if name == 'PC_VDISABLE':
            value = hex(value)
        print(FMT.format(name=name, value=value, description=description))
    print()


if __name__ == '__main__':
    display_fpathconf()
