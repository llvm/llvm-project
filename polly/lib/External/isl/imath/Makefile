##
## Name:     Makefile
## Purpose:  Makefile for imath library and associated tools
## Author:   M. J. Fromberger
##
## Copyright (C) 2002-2008 Michael J. Fromberger, All Rights Reserved.
##
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to
## deal in the Software without restriction, including without limitation the
## rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
## sell copies of the Software, and to permit persons to whom the Software is
## furnished to do so, subject to the following conditions:
##
## The above copyright notice and this permission notice shall be included in
## all copies or substantial portions of the Software.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
## FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
## IN THE SOFTWARE.
##

# --- begin configuration section ---

## Generic settings for systems with GCC (default)
## To build with debugging, add DEBUG=Y on the "make" command line.
ifeq ($(origin CC),default)
CC=gcc
endif
CFLAGS+=-pedantic -Wall -Werror -Wextra -Wno-unused-parameter \
	-I. -std=c99 $(DFLAGS$(DEBUG))
CSFLAGS=$(CFLAGS) -fPIC
#LIBS=

# These are needed to build the GMP compatibility tests.
export CC CFLAGS

DFLAGS=-O3 -funroll-loops -finline-functions
DFLAGSN=$(DFLAGS)
DFLAGSY=-g -DDEBUG=1

# --- end of configuration section ---

TARGETS=bintest bug-swap imtest imtimer rtest
HDRS=imath.h imrat.h iprime.h imdrover.h rsamath.h gmp_compat.h
SRCS=$(HDRS:.h=.c) $(TARGETS:=.c)
OBJS=$(SRCS:.c=.o)
OTHER=LICENSE ChangeLog Makefile doc.md doc.md.in \
	tools/findthreshold.py tools/mkdoc.py .dockerignore
VPATH += examples tests
EXAMPLES=basecvt findprime imcalc input pi randprime rounding rsakey

.PHONY: all test clean distclean
.SUFFIXES: .so .md

.c.o:
	$(CC) $(CFLAGS) -c $<

.c.so:
	$(CC) $(CSFLAGS) -o $@ -c $<

all: objs examples test

objs: $(OBJS)

check: test gmp-compat-test
	@ echo "Completed running imath and gmp-compat unit tests"

test: imtest pi bug-swap doc.md
	@ echo ""
	@ echo "Running tests, you should not see any 'FAILED' lines here."
	@ echo "If you do, please see doc.txt for how to report a bug."
	@ echo ""
	(cd tests && ./test.sh)

gmp-compat-test: libimath.so
	@ echo "Running gmp-compat unit tests"
	@ echo "Printing progress after every 100,000 tests"
	make -C tests/gmp-compat-test TESTS="-p 100000 random.tests"

docker-test:
	if which docker ; \
	then \
		docker run --rm -it \
		"$(shell docker build -f tests/linux/Dockerfile -q .)" ; \
	fi

$(EXAMPLES):%: imath.o imrat.o iprime.o %.o
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

$(TARGETS):%: imath.o %.o
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

examples: $(EXAMPLES)

libimath.so: imath.so imrat.so gmp_compat.so
	$(CC) $(CFLAGS) -shared -o $@ $^

imtest: imtest.o imath.o imrat.o imdrover.o iprime.o

rtest: rtest.o imath.o rsamath.o

# Requires clang-format: https://clang.llvm.org/docs/ClangFormat.html
format-c:
	@ echo "Formatting C source files and headers ..."
	find . -type f -name '*.h' -o -name '*.c' -print0 | \
		xargs -0 clang-format --style=Google -i

# Requires yapf: pip install yapf
format-py:
	@ echo "Formatting Python source files ..."
	find . -type f -name '*.py' -print0 | \
		xargs -0 yapf --style=pep8 -i

# Format source files.
format: format-c format-py

# Generate documentation from header comments.
# This rule depends on the header files to ensure the docs get updated when the
# headers change.
doc.md: doc.md.in imath.h imrat.h tools/mkdoc.py
	tools/mkdoc.py $< $@

clean distclean:
	rm -f *.o *.so *.pyc *~ core gmon.out tests/*~ tests/gmon.out examples/*~
	make -C tests/gmp-compat-test clean
	rm -f $(TARGETS) $(EXAMPLES)
