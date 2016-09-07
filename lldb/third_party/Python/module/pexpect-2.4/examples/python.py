#!/usr/bin/env python

"""This starts the python interpreter; captures the startup message; then gives
the user interactive control over the session. Why? For fun... """

# Don't do this unless you like being John Malkovich
# c = pexpect.spawn ('/usr/bin/env python ./python.py')

import pexpect
c = pexpect.spawn('/usr/bin/env python')
c.expect('>>>')
print 'And now for something completely different...'
f = lambda s: s and f(s[1:]) + s[0]  # Makes a function to reverse a string.
print f(c.before)
print 'Yes, it\'s python, but it\'s backwards.'
print
print 'Escape character is \'^]\'.'
print c.after,
c.interact()
c.kill(1)
print 'is alive:', c.isalive()
