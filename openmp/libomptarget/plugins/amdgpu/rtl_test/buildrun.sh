CLOC_PATH=/usr/bin
AOMP=${AOMP:-/opt/rocm/aomp}
AOMPRT_REPOS=${AOMPRT_REPOS:-$HOME/git/aomp}
RT_REPO_NAME=${RT_REPO_NAME:-openmp}

$CLOC_PATH/cloc.sh -ll -vv -opt 2  hw.cl

g++ rtl_test.cpp -lelf -L/usr/lib/x86_64-linux-gnu -lomptarget -lpthread -L${AOMP}/lib -I$AOMPRT_REPOS/$RT_REPO_NAME/libamdgcn}/src -L/opt/rocm/lib -lhsa-runtime64 -g -o rtl_test

LD_LIBRARY_PATH=/opt/rocm/lib:$AOMP/lib:$LD_LIBRARY_PATH ./rtl_test hw.hsaco

