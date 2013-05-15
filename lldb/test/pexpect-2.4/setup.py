'''
$Revision: 485 $
$Date: 2007-07-12 15:23:15 -0700 (Thu, 12 Jul 2007) $
'''
from distutils.core import setup
setup (name='pexpect',
    version='2.4',
    py_modules=['pexpect', 'pxssh', 'fdpexpect', 'FSM', 'screen', 'ANSI'],
    description='Pexpect is a pure Python Expect. It allows easy control of other applications.',
    author='Noah Spurrier',
    author_email='noah@noah.org',
    url='http://pexpect.sourceforge.net/',
    license='MIT license',
    platforms='UNIX',
)

#    classifiers = [
#        'Development Status :: 4 - Beta',
#        'Environment :: Console',
#        'Environment :: Console (Text Based)',
#        'Intended Audience :: Developers',
#        'Intended Audience :: System Administrators',
#        'Intended Audience :: Quality Engineers',
#        'License :: OSI Approved :: Python Software Foundation License',
#        'Operating System :: POSIX',
#        'Operating System :: MacOS :: MacOS X',
#        'Programming Language :: Python',
#        'Topic :: Software Development',
#        'Topic :: Software Development :: Libraries :: Python Modules',
#        'Topic :: Software Development :: Quality Assurance',
#        'Topic :: Software Development :: Testing',
#        'Topic :: System, System :: Archiving :: Packaging, System :: Installation/Setup',
#        'Topic :: System :: Shells',
#        'Topic :: System :: Software Distribution',
#        'Topic :: Terminals, Utilities',
#    ],



