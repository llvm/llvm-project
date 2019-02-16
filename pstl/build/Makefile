#===-- Makefile ----------------------------------------------------------===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is dual licensed under the MIT and the University of Illinois Open
# Source Licenses. See LICENSE.TXT for details.
#
#===----------------------------------------------------------------------===##

#------------------------------------------------------------------------------
# Define rules for making the Parallel STL library.
#------------------------------------------------------------------------------


proj_root ?= $(CURDIR)/..

include $(proj_root)/build/Makefile.common
CPLUS := $(compiler) $(COMPILER_NOLOGO_KEY) $(USE_SHARED_CPPRUNTIME_KEY)

.SECONDARY:
.PHONY: clean clean_all clean_pstl bench

VPATH = $(proj_root)/test

CPLUS_FLAGS += -I$(proj_root)

PSTL_MAKEFILE = $(proj_root)/build/Makefile.pstl
BENCH_MAKEFILE = $(proj_root)/build/Makefile.bench


test_hdr = $(wildcard $(proj_root)/test/*.h)
test_src = $(wildcard $(proj_root)/test/test_*.cpp)
test_bin = $(notdir $(test_src:.cpp=.exe))


all: $(test_bin)

test_%.offload.exe: test_%.offload$(OBJ_SFX) exception_list.offload$(OBJ_SFX)
	$(CPLUS) $(CPLUS_FLAGS) $^ $(FKEY)o$@ $(LDFLAGS)

$(PSTL_LIB_NAME):
	$(MAKE) -f $(PSTL_MAKEFILE) backend=$(backend) cfg=$(cfg)

test_%.exe: test_%$(OBJ_SFX) $(PSTL_LIB_NAME)
	$(LD) $< $(LD_OUT_KEY)$@ $(LDFLAGS) $(DYN_LDFLAGS) $(PSTL_LIB_LINK)

test_%: test_%.exe
	$(run_cmd) $(RUN_CMD)test_$*.exe

test_%$(OBJ_SFX): test_%.cpp $(test_hdr) $(proj_root)/build/Makefile
	$(CPLUS) $(CPLUS_FLAGS) -c $< $(FKEY)o$@

# This definition intentionally consists of two blank lines
define eol


endef

test: $(test_bin)
	$(foreach test, $(test_bin), $(run_cmd) $(RUN_CMD)$(test) $(args) $(eol))

%.s: %.cpp $(proj_root)/build/Makefile
	$(CPLUS) $(CPLUS_FLAGS) -S $< $(FKEY)o$@

%.E: %.cpp
	$(CPLUS) $(CPLUS_FLAGS) -E $< >$@

TEMPLATE_FILES=$(wildcard $(proj_root)/bench/*.*tmpl)
BENCH_COMMON_FILES=$(wildcard $(proj_root)/bench/*.h) $(wildcard $(proj_root)/bench/*.cpp)

$(BENCH_MAKEFILE): $(proj_root)/bench/algorithm.json $(proj_root)/bench/gen.py  $(TEMPLATE_FILES)
	$(PYTHON) $(proj_root)/bench/gen.py $(proj_root)/bench/algorithm.json

bench : $(BENCH_MAKEFILE) $(BENCH_COMMON_FILES) $(PSTL_LIB_NAME)
	@echo     TEMPLATE_FILES=$(TEMPLATE_FILES)
	@echo     proj_root=$(proj_root)
	ls -la      $(proj_root)/bench/gen.py
	$(MAKE) -f $(BENCH_MAKEFILE)

clean_bench:
	$(DEL_CMD) $(BENCH_MAKEFILE)
	$(DEL_CMD) batch.py
	$(DEL_CMD) $(proj_root)/build/bench/*.*

clean:
	$(DEL_CMD) *$(OBJ_SFX) *.exe *.E *.s *.asm *.d *.pdb *.pdb *.suo *.ilk

clean_pstl:
	$(MAKE) -f $(PSTL_MAKEFILE) clean

clean_all: clean clean_pstl clean_bench

info:
	@echo OS = $(os_name)
	@echo proj_root = "$(proj_root)"
	@echo $(CURDIR)
	@echo VPATH = $(VPATH)
	@echo LIBRARY_PATH = $(LIBRARY_PATH)
	@echo backend = $(backend)
	@echo compiler = $(compiler)

-include *.d
