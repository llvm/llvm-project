# This is for developers' convenience and not expected to be in build steps.
#
# Usage:
#   cd /path/to/llvm-project/llvm/test/tools/llvm-cov/Inputs
#   PATH=/path/to/build/bin:$PATH make -f yaml.makefile

CFLAGS_COVMAP	= -fcoverage-compilation-dir=. \
		  -mllvm -runtime-counter-relocation=true \
		  -mllvm -conditional-counter-update=true \
		  -mllvm -enable-name-compression=false \
		  -fprofile-instr-generate -fcoverage-mapping \
		  $(if $(filter mcdc-%, $*), $(CFLAGS_MCDC))

CFLAGS_MCDC	= -fcoverage-mcdc

%.o: %.cpp
	clang++ $< -c -o $@ $(CFLAGS_COVMAP)

%.o: %.c
	clang $< -c -o $@ $(CFLAGS_COVMAP)

%-single.o: %.cpp
	clang++ $< -c -o $@ \
		-mllvm -enable-single-byte-coverage=true \
		$(CFLAGS_COVMAP)

%-single.o: %.c
	clang $< -c -o $@ \
		-mllvm -enable-single-byte-coverage=true \
		$(CFLAGS_COVMAP)

%.covmap.o: %.o
	llvm-objcopy \
		--only-section=__llvm_covfun \
		--only-section=__llvm_covmap \
		--only-section=__llvm_prf_names \
		--strip-unneeded \
		$< $@

%.yaml: %.covmap.o
	obj2yaml $< > $@

%.exe: %.o
	clang++ -fprofile-instr-generate $^ -o $@

ARGS_branch-logical-mixed := \
	0 0; \
	0 1; \
	1 0; \
	1 1

ARGS_branch-macros := \
	0 1; \
	1 0; \
	1 1

ARGS_branch-showBranchPercentage := \
	0 1; \
	1 1; \
	2 2; \
	4 0; \
	5 0; \
	1

ARGS_showLineExecutionCounts := $(patsubst %,%;,$(shell seq 161))

ARGS_mcdc-const-folding := \
	0 1; \
	1 0; \
	1 1; \
	1 1

%.profdata: %.exe
	-find -name '$*.*.profraw' | xargs rm -fv
	@if [ "$(ARGS_$(patsubst %-single,%,$*))" = "" ]; then \
	  echo "Executing: $<"; \
	  LLVM_PROFILE_FILE=$*.%p%c.profraw ./$<; \
	else \
	  LLVM_PROFILE_FILE=$*.%p%c.profraw; \
	  export LLVM_PROFILE_FILE; \
	  for xcmd in $(shell echo "$(ARGS_$(patsubst %-single,%,$*))" | tr ';[:blank:]' ' %'); do \
	    cmd=$$(echo "$$xcmd" | tr '%' ' '); \
	    echo "Executing series: $< $$cmd"; \
	    eval "./$< $$cmd"; \
	  done; \
	fi
	find -name '$*.*.profraw' | xargs llvm-profdata merge --sparse -o $@

%.proftext: %.profdata
	llvm-profdata merge --text -o $@ $<

.PHONY: all
all:	\
	$(patsubst %.yaml,%.proftext, $(wildcard *.yaml)) \
	$(wildcard *.yaml)
	-find -name '*.profraw' | xargs rm -f
