// RUN: not llvm-tblgen -gen-runtime-libcalls -I %p/../../include -DERR_FN_ATTR %s 2>&1 | FileCheck -check-prefix=ERR-FN-ATTR %s
// RUN: not llvm-tblgen -gen-runtime-libcalls -I %p/../../include -DERR_PARAM_ATTR %s 2>&1 | FileCheck -check-prefix=ERR-PARAM-ATTR %s
// RUN: not llvm-tblgen -gen-runtime-libcalls -I %p/../../include -DERR_RET_ATTR %s 2>&1 | FileCheck -check-prefix=ERR-RET-ATTR %s
// RUN: not llvm-tblgen -gen-runtime-libcalls -I %p/../../include -DERR_INT_ATTR %s 2>&1 | FileCheck -check-prefix=ERR-INT-ATTR %s
// RUN: not llvm-tblgen -gen-runtime-libcalls -I %p/../../include -DERR_TOO_MANY_PARAM_ATTRS %s 2>&1 | FileCheck -check-prefix=ERR-TOO-MANY-ATTRS %s
// RUN: not llvm-tblgen -gen-runtime-libcalls -I %p/../../include -DERR_TOO_MANY_POINTEE_TYPES %s 2>&1 | FileCheck -check-prefix=ERR-TOO-MANY-POINTEE %s
// RUN: not llvm-tblgen -gen-runtime-libcalls -I %p/../../include -DERR_POINTEE_NON_PTR %s 2>&1 | FileCheck -check-prefix=ERR-POINTEE-NON-PTR %s

include "llvm/IR/RuntimeLibcallsImpl.td"

#ifdef ERR_FN_ATTR
def MODF_F32 : RuntimeLibcall<Flt, [Flt, Ptr]> {
  let FnAttrs = [NoUnwind, NoAlias];
}

// ERR-FN-ATTR: [[@LINE+1]]:5: error: 'NoAlias' is not a valid FnAttr
def modff : RuntimeLibcallImpl<MODF_F32>;
#endif

#ifdef ERR_PARAM_ATTR
def MODF_F32 : RuntimeLibcall<Flt, [Flt, Ptr]> {
  let ParamAttrs = [[], [NoAlias, WillReturn]];
}

// ERR-PARAM-ATTR: [[@LINE+1]]:5: error: 'WillReturn' is not a valid ParamAttr
def modff : RuntimeLibcallImpl<MODF_F32>;
#endif

#ifdef ERR_RET_ATTR
def MODF_F32 : RuntimeLibcall<Flt, [Flt, Ptr]> {
  let RetAttrs = [NoUndef, WillReturn];
}

// ERR-RET-ATTR: [[@LINE+1]]:5: error: 'WillReturn' is not a valid RetAttr
def modff : RuntimeLibcallImpl<MODF_F32>;
#endif

#ifdef ERR_INT_ATTR
def MALLOC : RuntimeLibcall<Ptr, [SizeT]> {
  let FnAttrs = [NoUnwind, AllocKind];
}

// ERR-INT-ATTR: [[@LINE+1]]:5: error: 'AllocKind' requires a value; use a ValueAttr
def malloc : RuntimeLibcallImpl<MALLOC>;
#endif

#ifdef ERR_TOO_MANY_PARAM_ATTRS
def MODF_F32 : RuntimeLibcall<Flt, [Flt, Ptr]> {
  let ParamAttrs = [[], [NoAlias], [NonNull]];
}

// ERR-TOO-MANY-ATTRS: [[@LINE+1]]:5: error: ParamAttrs has 3 entries but the function has 2 arguments
def modff : RuntimeLibcallImpl<MODF_F32>;
#endif

#ifdef ERR_TOO_MANY_POINTEE_TYPES
def MODF_F32 : RuntimeLibcall<Flt, [Flt, Ptr]> {
  let ParamPointeeTypes = [NoneType, Flt, Flt];
}

// ERR-TOO-MANY-POINTEE: [[@LINE+1]]:5: error: ParamPointeeTypes has 3 entries but the function has 2 arguments
def modff : RuntimeLibcallImpl<MODF_F32>;
#endif

#ifdef ERR_POINTEE_NON_PTR
def MODF_F32 : RuntimeLibcall<Flt, [Flt, Ptr]> {
  let ParamPointeeTypes = [Flt, Flt];
}

// ERR-POINTEE-NON-PTR: [[@LINE+1]]:5: error: ParamPointeeTypes entry for non-pointer argument 0
def modff : RuntimeLibcallImpl<MODF_F32>;
#endif
