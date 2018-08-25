## Process this file with automake to produce Makefile.in.

AUTOMAKE_OPTIONS = foreign dejagnu

# Setup the testing framework, if you have one
EXPECT = `if [ -f $(top_builddir)/../expect/expect ] ; then \
            echo $(top_builddir)/../expect/expect ; \
          else echo expect ; fi`

RUNTEST = `if [ -f $(top_srcdir)/../dejagnu/runtest ] ; then \
	       echo $(top_srcdir)/../dejagnu/runtest ; \
	    else echo runtest; fi`

# When running the tests we set GCC_EXEC_PREFIX to the install tree so that
# files that have already been installed there will be found.  The -B option
# overrides it, so use of GCC_EXEC_PREFIX will not result in using GCC files
# from the install tree.

AM_RUNTESTFLAGS = "TEST_GCC_EXEC_PREFIX=$(libdir)/gcc"

CLEANFILES = *.log *.sum
