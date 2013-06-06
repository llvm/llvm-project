LEVEL := ..
include $(LEVEL)/Makefile.common

# Test in all immediate subdirectories if unset.
ifdef TESTSUITE
TESTDIRS := $(TESTSUITE:%=$(PROJ_SRC_DIR)/%)
else
TESTDIRS ?= $(PROJ_SRC_DIR)
endif

# 'lit' wants objdir paths, so it will pick up the lit.site.cfg.
TESTDIRS := $(TESTDIRS:$(PROJ_SRC_DIR)%=$(PROJ_OBJ_DIR)%)

# Allow EXTRA_TESTDIRS to provide additional test directories.
TESTDIRS += $(EXTRA_TESTDIRS)

ifndef TESTARGS
ifdef VERBOSE
TESTARGS = -v
else
TESTARGS = -s -v
endif
endif

# Make sure any extra test suites can find the main site config.
LIT_ARGS := --param polly_site_config=${PROJ_OBJ_DIR}/lit.site.cfg \
            --param build_config=$(PROJ_OBJ_DIR)


ifdef VG
  LIT_ARGS += "--vg"
endif


check-polly:: lit.site.cfg polly-lib
	@ echo '--- Running polly tests for $(TARGET_TRIPLE) ---'
	@ $(PYTHON) $(LLVM_SRC_ROOT)/utils/lit/lit.py \
	  $(LIT_ARGS) $(TESTARGS) $(TESTDIRS)

FORCE:

polly-lib: FORCE
	$(MAKE) -C ../lib || exit 1;

lit.site.cfg: FORCE
	@echo "Making Polly 'lit.site.cfg' file..."
	@sed -e "s#@LLVM_SOURCE_DIR@#$(LLVM_SRC_ROOT)#g" \
	     -e "s#@LLVM_BINARY_DIR@#$(LLVM_OBJ_ROOT)#g" \
	     -e "s#@LLVM_TOOLS_DIR@#$(LLVM_OBJ_ROOT)/$(BuildMode)/bin#g" \
	     -e "s#@LLVM_LIBS_DIR@#$(LibDir)#g" \
	     -e "s#@POLLY_SOURCE_DIR@#$(PROJ_SRC_DIR)/..#g" \
	     -e "s#@POLLY_BINARY_DIR@#$(PROJ_OBJ_DIR)/..#g" \
	     -e "s#@TARGET_TRIPLE@#$(TARGET_TRIPLE)#g" \
	     -e "s#@LLVM_SHLIBEXT@#$(SHLIBEXT)#g" \
             -e "s#@POLLY_LIB_DIR@#$(LibDir)#g" \
	     $(PROJ_SRC_DIR)/lit.site.cfg.in > $@

clean::
	@ find . -name Output | xargs rm -fr

.PHONY: check-polly clean
