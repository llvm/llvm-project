# This is for developers' convenience and not expected to be in build steps.
#
# Usage:
#   cd /path/to/llvm-project/llvm/test/tools/llvm-cov/Inputs
#   PATH=/path/to/build/bin:$PATH make -f yaml.makefile *.yaml

%.covmap.o: %.o
	llvm-objcopy \
		--only-section=__llvm_covfun \
		--only-section=__llvm_covmap \
		--only-section=__llvm_prf_names \
		--strip-unneeded \
		$< $@

%.yaml: %.covmap.o
	obj2yaml $< > $@
