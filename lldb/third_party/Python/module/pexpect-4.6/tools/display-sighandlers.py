#!/usr/bin/env python
# Displays all signals, their values, and their handlers.
from __future__ import print_function
import signal
FMT = '{name:<10} {value:<5} {description}'

# header
print(FMT.format(name='name', value='value', description='description'))
print('-' * (33))

for name, value in [(signal_name, getattr(signal, signal_name))
                    for signal_name in dir(signal)
                    if signal_name.startswith('SIG')
                    and not signal_name.startswith('SIG_')]:
    try:
        handler = signal.getsignal(value)
    except ValueError:
        # FreeBSD: signal number out of range
        handler = 'out of range'
    description = {
        signal.SIG_IGN: "ignored(SIG_IGN)",
        signal.SIG_DFL: "default(SIG_DFL)"
    }.get(handler, handler)
    print(FMT.format(name=name, value=value, description=description))
