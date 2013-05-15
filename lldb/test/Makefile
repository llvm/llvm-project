LLDB_LEVEL := ..
include $(LLDB_LEVEL)/Makefile

.PHONY: programs

all:: check-local

#----------------------------------------------------------------------
# Make all of the test programs
#----------------------------------------------------------------------
programs:
	find . -type d -depth 1 | xargs -J % find % \
		-name Makefile \
		-exec echo \; \
		-exec echo make -f '{}' \; \
		-execdir make \;

#----------------------------------------------------------------------
# Clean all of the test programs
#----------------------------------------------------------------------
clean::
	find . -type d -depth 1 | xargs -J % find % \
		-name Makefile \
		-exec echo \; \
		-exec echo make -f '{}' clean \; \
		-execdir make clean \;

#----------------------------------------------------------------------
# Run the tests
#----------------------------------------------------------------------
check-local::
	rm -rf lldb-test-traces
	python $(PROJ_SRC_DIR)/dosep.ty -o "--executable $(ToolDir)/lldb -q -s lldb-test-traces -u CXXFLAGS -u CFLAGS -C $(subst ccache,,$(CC))"
