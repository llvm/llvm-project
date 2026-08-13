// RUN: not llvm-tblgen -gen-runtime-libcalls -I %p/../../include -DERR_DUP_INTRINSIC %s 2>&1 | FileCheck -check-prefix=ERR-DUP %s
// RUN: not llvm-tblgen -gen-runtime-libcalls -I %p/../../include -DERR_NO_LIBCALL %s 2>&1 | FileCheck -check-prefix=ERR-NO-LIBCALL %s
// RUN: not llvm-tblgen -gen-runtime-libcalls -I %p/../../include -DERR_NO_LIBCALL_NO_INTRINSIC %s 2>&1 | FileCheck -check-prefix=ERR-NO-LIBCALL-NO-INTRIN %s

include "llvm/IR/RuntimeLibcallsImpl.td"

#ifdef ERR_DUP_INTRINSIC
def SIN_F32 : RuntimeLibcall;
def impl_sinf : RuntimeLibcallImpl<SIN_F32, "sinf">;
def COS_F32 : RuntimeLibcall;
def impl_cosf : RuntimeLibcallImpl<COS_F32, "cosf">;
def : RuntimeLibcallFamily<"SIN", ["sin"]>;
def : RuntimeLibcallFamily<"COS", ["sin"]>;

// ERR-DUP: error: intrinsic 'sin' is mapped by multiple RuntimeLibcallFamily records ('COS' and 'SIN')
#endif

#ifdef ERR_NO_LIBCALL
def COS_F32 : RuntimeLibcall;
def impl_cosf : RuntimeLibcallImpl<COS_F32, "cosf">;
def : RuntimeLibcallFamily<"SIN", ["sin"]>;

// ERR-NO-LIBCALL: error: no runtime libcall found for base name 'SIN'
#endif

#ifdef ERR_NO_LIBCALL_NO_INTRINSIC
def COS_F32 : RuntimeLibcall;
def impl_cosf : RuntimeLibcallImpl<COS_F32, "cosf">;
def : RuntimeLibcallFamily<"SIN">;

// ERR-NO-LIBCALL-NO-INTRIN: error: no runtime libcall found for base name 'SIN'
#endif
