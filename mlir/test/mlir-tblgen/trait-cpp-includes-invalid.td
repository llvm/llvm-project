// RUN: not mlir-tblgen -gen-typedef-decls -I %S/../../include -DERROR1 %s 2>&1 | FileCheck --check-prefix=ERROR1 %s
// RUN: not mlir-tblgen -gen-op-decls -I %S/../../include -DERROR2 %s 2>&1 | FileCheck --check-prefix=ERROR2 %s
// RUN: not mlir-tblgen -gen-attrdef-decls -I %S/../../include -DERROR3 %s 2>&1 | FileCheck --check-prefix=ERROR3 %s
// RUN: not mlir-tblgen -gen-op-decls -I %S/../../include -DERROR4 %s 2>&1 | FileCheck --check-prefix=ERROR4 %s
// RUN: not mlir-tblgen -gen-op-decls -I %S/../../include -DERROR5 %s 2>&1 | FileCheck --check-prefix=ERROR5 %s
// RUN: not mlir-tblgen -gen-op-interface-decls -I %S/../../include -DERROR6 %s 2>&1 | FileCheck --check-prefix=ERROR6 %s

// Headers are emitted verbatim in `#include "..."`. Anything that could end the
// header name or the line, or start a comment, is rejected so that no part of a
// header can escape the directive.

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/Interfaces.td"
include "mlir/IR/OpBase.td"

def Test_Dialect : Dialect {
  let name = "test";
  let cppNamespace = "::test";
}

#ifdef ERROR1
// ERROR1: error: cannot emit '#include' for header "test/Evil.h\"\nvoid injected();\n#include \"test/X.h": header must not contain '"'
def InjectionTypeTrait : NativeTypeTrait<"InjectionTypeTrait"> {
  let cppIncludes = ["test/Evil.h\"\nvoid injected();\n#include \"test/X.h"];
}
def TestType : TypeDef<Test_Dialect, "Test", [InjectionTypeTrait]> {
  let mnemonic = "test";
}
#endif

#ifdef ERROR2
// ERROR2: error: cannot emit '#include' for header "test\\A.h": header must not contain '\'
def BackslashOpTrait : NativeOpTrait<"BackslashOpTrait"> {
  let cppIncludes = ["test\\A.h"];
}
def TestOp : Op<Test_Dialect, "test", [BackslashOpTrait]>;
#endif

#ifdef ERROR3
// ERROR3: error: cannot emit '#include' for header "test/*/A.h": header must not contain a comment
def BlockCommentAttrTrait : NativeAttrTrait<"BlockCommentAttrTrait"> {
  let cppIncludes = ["test/*/A.h"];
}
def TestAttr : AttrDef<Test_Dialect, "Test", [BlockCommentAttrTrait]> {
  let mnemonic = "test";
}
#endif

#ifdef ERROR4
// ERROR4: error: cannot emit '#include' for header "test//A.h": header must not contain a comment
def LineCommentOpTrait : NativeOpTrait<"LineCommentOpTrait"> {
  let cppIncludes = ["test//A.h"];
}
def TestOp : Op<Test_Dialect, "test", [LineCommentOpTrait]>;
#endif

#ifdef ERROR5
// ERROR5: error: cannot emit '#include' for header "test/A.h\n": header must not contain non-printable characters
def NewlineOpTrait : NativeOpTrait<"NewlineOpTrait"> {
  let cppIncludes = ["test/A.h\n"];
}
def TestOp : Op<Test_Dialect, "test", [NewlineOpTrait]>;
#endif

#ifdef ERROR6
// ERROR6: error: cannot emit '#include' for header "test/A.h\"": header must not contain '"'
def BaseOpInterface : OpInterface<"BaseOpInterface"> {
  let cppNamespace = "::test";
  let cppIncludes = ["test/A.h\""];
}
def DerivedOpInterface : OpInterface<"DerivedOpInterface", [BaseOpInterface]> {
  let cppNamespace = "::test";
}
#endif
