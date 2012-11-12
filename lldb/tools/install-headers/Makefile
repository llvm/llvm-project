installsrc:
	echo "installsrc (doing nothing)"

install:
	echo "install (doing nothing)"

clean:
	echo "clean (doing nothing)"

TRUNCATED_VERSION = $(shell echo $(CURRENT_PROJECT_VERSION) | /usr/bin/sed -E "s/^([0-9]+)(\.[0-9]+)?$$/\1/g")

installhdrs:
	cd "${TARGET_BUILD_DIR}/${LLDB_FRAMEWORK_INSTALL_DIR}/LLDB.framework/Headers" ;\
	for file in *.h ;\
	do \
		/usr/bin/sed -i '' 's/\(#include\)[ ]*"lldb\/\(API\/\)\{0,1\}\(.*\)"/\1 <LLDB\/\3>/1' "$$file" ;\
		/usr/bin/sed -i '' 's|<LLDB/Utility|<LLDB|' "$$file" ;\
		/usr/bin/sed -i '' "s|//#define LLDB_VERSION|#define LLDB_VERSION $(TRUNCATED_VERSION) |" "$$file" ;\
	done
