installsrc:
	echo "installsrc (doing nothing)"

install:
	echo "install (doing nothing)"

clean:
	echo "clean (doing nothing)"

LLDB_VERSION=`echo ${CURRENT_PROJECT_VERSION} | /usr/bin/sed -E 's/^([0-9]+).([0-9]+).([0-9]+)(.[0-9]+)?$/\1/g'`
LLDB_REVISION=`echo ${CURRENT_PROJECT_VERSION} | /usr/bin/sed -E 's/^([0-9]+).([0-9]+).([0-9]+)(.[0-9]+)?$/\3/g'`
LLDB_VERSION_STRING=`echo ${CURRENT_PROJECT_VERSION}`

installhdrs:
	cd "${TARGET_BUILD_DIR}/${LLDB_FRAMEWORK_INSTALL_DIR}/LLDB.framework/Headers" ;\
	for file in *.h ;\
	do \
		/usr/bin/sed -i '' 's/\(#include\)[ ]*"lldb\/\(API\/\)\{0,1\}\(.*\)"/\1 <LLDB\/\3>/1' "$$file" ;\
		/usr/bin/sed -i '' 's|<LLDB/Utility|<LLDB|' "$$file" ;\
		/usr/bin/sed -i '' "s|//#define LLDB_VERSION$|#define LLDB_VERSION $(LLDB_VERSION) |" "$$file" ;\
		/usr/bin/sed -i '' "s|//#define LLDB_REVISION|#define LLDB_REVISION $(LLDB_REVISION) |" "$$file" ;\
		/usr/bin/sed -i '' "s|//#define LLDB_VERSION_STRING|#define LLDB_VERSION_STRING \"$(LLDB_VERSION_STRING)\" |" "$$file" ;\
	done
