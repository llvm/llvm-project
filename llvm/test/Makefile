LEVEL  = ..
DIRS   = Programs

#
# Make QMTest the default for testing features and regressions
# Do this first to force QMTest to run first
#
all:: qmtest

#
# Include other test rules
#
include Makefile.tests

#
# New QMTest functionality:
#	The test suite is being transitioned over to QMTest.  Eventually, it
#	will use QMTest by default.
#

# QMTest option specifying the location of the QMTest database.
QMDB= -D $(LLVM_SRC_ROOT)/test
QMCLASSES=$(LLVM_OBJ_ROOT)/test/QMTest

#
# Determine which expectations file we will use
#
QMEXPECT:=$(LLVM_SRC_ROOT)/test/QMTest/expectations.unknown.qmr
ifeq ($(OS),Linux)
QMEXPECT:=$(LLVM_SRC_ROOT)/test/QMTest/expectations.linux.qmr
endif

ifeq ($(OS),SunOS)
QMEXPECT:=$(LLVM_SRC_ROOT)/test/QMTest/expectations.sunos.qmr
endif

ifeq ($(OS),Darwin)
QMEXPECT:=$(LLVM_SRC_ROOT)/test/QMTest/expectations.darwin.qmr
endif

#
# This is configuration information used by the test suite.  In QM Test, it's
# called a 'context.'
#
CONTEXT= -c "srcroot=$(LLVM_SRC_ROOT)" \
         -c "buildroot=$(LLVM_OBJ_ROOT)" \
         -c "buildtype=$(CONFIGURATION)" \
         -c "tmpdir=$(LLVM_OBJ_ROOT)/test/tmp" \
         -c "coresize=0" \
         -c "cc=$(CC)" \
         -c "cxx=$(CXX)" \
         -c "llvmgcc=$(LLVMGCC)" \
         -c "llvmgxx=$(LLVMGXX)" \
         -c "make=$(MAKE)"

#
# Location of the QMTest program.
#
QMTEST= QMTEST_CLASS_PATH=$(QMCLASSES) qmtest $(QMDB)


#
# Execute the tests
#
qmtest:: $(LLVM_OBJ_ROOT)/test/tmp register
	-$(QMTEST) run -O $(QMEXPECT) $(CONTEXT)

%.t:: $(LLVM_OBJ_ROOT)/test/tmp register
	-$(QMTEST) run -O $(QMEXPECT) $(CONTEXT) $*

#
# Create the temporary directory used by the test suite.
#
$(LLVM_OBJ_ROOT)/test/tmp::
	${MKDIR} $(LLVM_OBJ_ROOT)/test/tmp

#
# Right now, QMTest compiles the python test classes and put them into the
# source tree.  Since Python bytecode is *not* cross-platform compatible (I
# think), we'll regenerate every time.
#
# Simultaneous builds won't work, but shared source trees will.
#
register:
	$(QMTEST) register test llvm.TestAsmDisasm
	$(QMTEST) register test llvm.AssembleTest
	$(QMTEST) register test llvm.ConvertToCTest
	$(QMTEST) register test llvm.LLToCTest
	$(QMTEST) register test llvm.MachineCodeTest
	$(QMTEST) register test llvm.TestOptimizer
	$(QMTEST) register test llvm.LLITest
	$(QMTEST) register test llvm.TestRunner
	$(QMTEST) register test llvm.VerifierTest
	$(QMTEST) register test llvm.CTest
	$(QMTEST) register test llvm.CXXTest
	$(QMTEST) register database llvmdb.llvmdb

# Start up the QMTest GUI
gui::
	$(QMTEST) gui --no-browser --daemon

# Also get rid of qmtest garbage when we 'make clean' in this directory.
clean:: qmtest-clean

qmtest-clean:
	$(RM) -rf $(LLVM_OBJ_ROOT)/test/tmp
	$(RM) -f $(LLVM_SRC_ROOT)/test/QMTest/*.pyo \
		$(LLVM_OBJ_ROOT)/test/QMTest/*.pyo
	$(RM) -f $(LLVM_SRC_ROOT)/test/results.qmr \
		$(LLVM_OBJ_ROOT)/test/results.qmr

