This directory contains Python bindings for LLVM's C library.

The bindings are currently a work in progress and are far from complete.
Use at your own risk.

Developer Info
==============

The single Python package is "llvm." Modules inside this package roughly
follow the names of the modules/headers defined by LLVM's C API.

Testing
-------

All test code is location in llvm/tests. Tests are written as classes
which inherit from llvm.tests.base.TestBase, which is a convenience base
class that provides common functionality.

Tests can be executed by installing nose:

    pip install nosetests

Then by running nosetests:

    nosetests

To see more output:

    nosetests -v

To step into the Python debugger while running a test, add the following
to your test at the point you wish to enter the debugger:

    import pdb; pdb.set_trace()

Then run nosetests:

    nosetests -s -v

You should strive for high code coverage. To see current coverage:

    pip install coverage
    nosetests --with-coverage --cover-html

Then open cover/index.html in your browser of choice to see the code coverage.

Style Convention
----------------

All code should pass PyFlakes. First, install PyFlakes:

    pip install pyflakes

Then at any time run it to see a report:

    pyflakes .

Eventually we'll provide a Pylint config file. In the meantime, install
Pylint:

    pip install pylint

And run:

    pylint llvm

And try to keep the number of violations to a minimum.
